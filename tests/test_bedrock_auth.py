"""
Test AWS Bedrock authentication and model access.
This script verifies that we can connect to AWS Bedrock and use the Nova Lite model.
"""

import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError

def test_bedrock_auth():
    """Test basic Bedrock authentication and model access."""

    print("🔍 Testing AWS Bedrock Authentication...")
    print("-" * 60)

    # 1. Check AWS credentials
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS Authentication successful!")
        print(f"   Account: {identity['Account']}")
        print(f"   ARN: {identity['Arn']}")
        print(f"   User ID: {identity['UserId']}")
    except NoCredentialsError:
        print("❌ No AWS credentials found. Please configure AWS credentials.")
        return False
    except ClientError as e:
        print(f"❌ AWS authentication failed: {e}")
        return False

    print()

    # 2. Check Bedrock access
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
        print("✅ Bedrock client created successfully (region: us-west-2)")
    except Exception as e:
        print(f"❌ Failed to create Bedrock client: {e}")
        return False

    print()

    # 3. Test model invocation with Nova Lite
    model_id = "us.amazon.nova-lite-v1:0"
    print(f"🧪 Testing model invocation: {model_id}")

    try:
        # Create a simple test request
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Hello, this is a test. Please respond with 'Hello back!'"}]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        }

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())

        # Extract the response text (Nova Lite format)
        if 'output' in response_body and 'message' in response_body['output']:
            response_text = response_body['output']['message']['content'][0]['text']
            print(f"✅ Model invocation successful!")
            print(f"   Response: {response_text}")
            return True
        else:
            print(f"⚠️  Unexpected response format: {response_body}")
            return False

    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ Model invocation failed:")
        print(f"   Error Code: {error_code}")
        print(f"   Error Message: {error_message}")

        if error_code == 'AccessDeniedException':
            print("\n💡 Troubleshooting:")
            print("   - Verify your IAM role has bedrock:InvokeModel permissions")
            print("   - Check if model access is enabled in Bedrock console")
            print("   - Ensure you're using the correct AWS region (us-west-2)")

        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_bedrock_auth()

    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed! Bedrock is ready to use.")
    else:
        print("❌ Some tests failed. Please resolve the issues above.")
    print("=" * 60)
