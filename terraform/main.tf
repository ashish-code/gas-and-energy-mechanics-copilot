provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  image_uri  = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  tags = {
    project = var.service_name
    owner   = "ashish-code"
  }
}

# ---------------------------------------------------------------------------
# ECR — persists between deploy/teardown cycles so you don't need to rebuild
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = var.service_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true  # allows destroy even when images are present

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = local.tags
}

# Expire untagged images after 14 days to keep storage costs minimal
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM — task role (Bedrock + CloudWatch) and ECR access role (data source)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task_role" {
  name = "gas-energy-copilot-apprunner-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "tasks.apprunner.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "task_policy" {
  name = "gas-energy-copilot-bedrock-policy"
  role = aws_iam_role.task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "BedrockInvokeGPT"
        Effect = "Allow"
        Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/openai.gpt-oss-120b-1:0",
          "arn:aws:bedrock:${var.aws_region}::foundation-model/us.openai.gpt-oss-120b-1:0",
        ]
      },
      {
        Sid      = "BedrockInvokeTitanEmbeddings"
        Effect   = "Allow"
        Action   = "bedrock:InvokeModel"
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/apprunner/*"
      },
    ]
  })
}

# The App Runner ECR access role already exists in every AWS account that
# has used App Runner. Use a data source so Terraform doesn't own it.
data "aws_iam_role" "ecr_access_role" {
  name = "AppRunnerECRAccessRole"
}

# ---------------------------------------------------------------------------
# App Runner service
# ---------------------------------------------------------------------------

resource "aws_apprunner_service" "app" {
  service_name = var.service_name

  source_configuration {
    authentication_configuration {
      access_role_arn = data.aws_iam_role.ecr_access_role.arn
    }

    image_repository {
      image_identifier      = local.image_uri
      image_repository_type = "ECR"

      image_configuration {
        port = "8080"
        runtime_environment_variables = {
          AWS_REGION = var.aws_region
          CONFIG_DIR = "/app/config"
        }
      }
    }

    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = tostring(var.cpu)
    memory            = tostring(var.memory)
    instance_role_arn = aws_iam_role.task_role.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/health"
    interval            = 20
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 3
  }

  tags = local.tags

  depends_on = [aws_iam_role_policy.task_policy]
}
