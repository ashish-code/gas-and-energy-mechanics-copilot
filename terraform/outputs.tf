output "service_url" {
  description = "Public HTTPS URL of the App Runner service"
  value       = "https://${aws_apprunner_service.app.service_url}"
}

output "ecr_repository_url" {
  description = "ECR repository URL (use this when tagging images)"
  value       = aws_ecr_repository.app.repository_url
}

output "service_arn" {
  description = "App Runner service ARN"
  value       = aws_apprunner_service.app.arn
}

output "task_role_arn" {
  description = "IAM task role ARN"
  value       = aws_iam_role.task_role.arn
}
