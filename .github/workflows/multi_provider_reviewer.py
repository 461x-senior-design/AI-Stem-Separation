#!/usr/bin/env python3
"""
Multi-provider AI code review script.
Supports Claude, OpenAI GPT, and Qwen models.
"""

import os
import sys
import json

def get_ai_provider():
    """Determine which AI provider to use based on available API keys."""
    if os.environ.get('ANTHROPIC_API_KEY'):
        return 'claude'
    elif os.environ.get('OPENAI_API_KEY'):
        return 'openai'
    elif os.environ.get('QWEN_API_KEY'):
        return 'qwen'
    else:
        return None

def review_with_claude(diff_content, changed_files):
    """Review code using Anthropic Claude."""
    from anthropic import Anthropic
    
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    client = Anthropic(api_key=api_key)
    
    prompt = f"""You are an expert code reviewer. Review the following PR changes:

Changed Files:
{changed_files}

Diff:
{diff_content}

Provide detailed feedback on code quality, bugs, security, and improvements."""
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

def review_with_openai(diff_content, changed_files):
    """Review code using OpenAI GPT."""
    import openai
    
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    
    prompt = f"""You are an expert code reviewer. Review the following PR changes:

Changed Files:
{changed_files}

Diff:
{diff_content}

Provide detailed feedback on code quality, bugs, security, and improvements."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an expert code reviewer."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096
    )
    
    return response.choices[0].message.content

def review_with_qwen(diff_content, changed_files):
    """Review code using Qwen (Alibaba Cloud)."""
    # Note: This is a placeholder. Actual implementation depends on Qwen API client
    # You would need to install and configure the Qwen SDK
    # pip install dashscope
    
    print("Qwen integration is not yet implemented.")
    print("To add Qwen support:")
    print("1. Install dashscope: pip install dashscope")
    print("2. Get API key from Alibaba Cloud")
    print("3. Implement the Qwen API call here")
    
    return "Qwen review not available. Please configure Qwen API or use Claude/OpenAI."

def main():
    """Main function to route to appropriate AI provider."""
    provider = get_ai_provider()
    
    if not provider:
        print("ERROR: No AI API key found!")
        print("Please set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or QWEN_API_KEY")
        sys.exit(1)
    
    print(f"Using AI provider: {provider.upper()}")
    
    # Read diff and files
    with open('pr_diff.txt', 'r') as f:
        diff_content = f.read()
    with open('changed_files.txt', 'r') as f:
        changed_files = f.read()
    
    # Get review based on provider
    try:
        if provider == 'claude':
            review = review_with_claude(diff_content, changed_files)
        elif provider == 'openai':
            review = review_with_openai(diff_content, changed_files)
        elif provider == 'qwen':
            review = review_with_qwen(diff_content, changed_files)
        else:
            print(f"Unknown provider: {provider}")
            sys.exit(1)
        
        # Save and display review
        with open('review_output.txt', 'w') as f:
            f.write(review)
        
        print("\n" + "="*80)
        print(f"AI Review ({provider.upper()}):")
        print("="*80)
        print(review)
        print("="*80)
        
        return 0
    except Exception as e:
        print(f"Error during review: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
