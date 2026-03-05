variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use for authentication"
  type        = string
  default     = "vscode-user"
}

variable "service_name" {
  description = "Name used for the App Runner service, ECR repo, and IAM role prefix"
  type        = string
  default     = "gas-and-energy-mechanics-copilot"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "cpu" {
  description = "App Runner vCPU allocation (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 1024  # 1 vCPU
}

variable "memory" {
  description = "App Runner memory allocation in MB (512, 1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288)"
  type        = number
  default     = 2048  # 2 GB
}
