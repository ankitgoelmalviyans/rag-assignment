# Dummy infrastructure for the demo: a bucket holding the RAG corpus.
# Shows the shape the policy in ../policy/terraform.rego expects.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-1"
}

resource "aws_s3_bucket" "rag_corpus" {
  bucket = "acme-rag-corpus-demo"

  tags = {
    owner          = "platform-team"
    classification = "confidential"
    cost_center    = "ai-platform"
  }
}

resource "aws_s3_bucket_public_access_block" "rag_corpus" {
  bucket                  = aws_s3_bucket.rag_corpus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "rag_corpus" {
  bucket = aws_s3_bucket.rag_corpus.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}
