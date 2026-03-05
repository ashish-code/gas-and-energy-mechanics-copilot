terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # State is stored locally. Keep terraform.tfstate out of git (see .gitignore).
  # To use an S3 backend instead, replace this block with:
  #   backend "s3" {
  #     bucket  = "your-tfstate-bucket"
  #     key     = "gas-energy-copilot/terraform.tfstate"
  #     region  = "us-east-1"
  #     profile = "vscode-user"
  #   }
}
