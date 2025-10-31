# AI Code Review Setup Guide

## Overview

This repository includes automated AI-powered code reviews for pull requests. The system currently supports:

- **Claude (Anthropic)** - Primary implementation ✅
- **OpenAI GPT** - Extension ready 🔧
- **Qwen (Alibaba)** - Extension ready 🔧

## Quick Start

### Step 1: Choose Your AI Provider

#### Option A: Claude (Recommended - Already Configured)

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account and get an API key
3. Go to your GitHub repository → Settings → Secrets and variables → Actions
4. Add a new secret:
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your Anthropic API key

#### Option B: OpenAI GPT

1. Get an API key from [OpenAI](https://platform.openai.com/)
2. Add to repository secrets:
   - Name: `OPENAI_API_KEY`
   - Value: Your OpenAI API key
3. Modify `.github/workflows/ai-pr-review.yml`:
   - Update the install dependencies step to include: `pip install openai`
   - Change the script to use `multi_provider_reviewer.py`

#### Option C: Qwen (Alibaba Cloud)

1. Get an API key from [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
2. Add to repository secrets:
   - Name: `QWEN_API_KEY`
   - Value: Your Qwen API key
3. Implement the Qwen integration in `multi_provider_reviewer.py`
4. Update workflow to use the multi-provider script

### Step 2: Test the Integration

1. Create a test branch
2. Make a small code change
3. Open a pull request
4. Wait for the AI review workflow to run
5. Check the PR comments for AI feedback

## Advanced Configuration

### Customizing Review Behavior

Edit `.github/ai-review-config.yml`:

```yaml
AI_MODEL: "claude-3-5-sonnet-20241022"  # Change model
VERBOSITY: "high"  # low, medium, high
MAX_TOKENS: 4096  # Response length limit
```

### Switching AI Providers

To use multiple providers or switch between them:

1. Add API keys for multiple providers as repository secrets
2. Update the workflow to use `multi_provider_reviewer.py`
3. The script will automatically use the first available API key

### Excluding Files from Review

Add patterns to `.github/ai-review-config.yml`:

```yaml
EXCLUDE_PATTERNS:
  - "*.md"
  - "docs/**"
  - "test/fixtures/**"
```

## Troubleshooting

### Workflow Fails with "API Key Not Found"

- Verify the secret is named exactly `ANTHROPIC_API_KEY` (or your chosen provider)
- Check that the secret is available in repository settings
- Ensure the secret has no extra spaces or characters

### Review Comments Not Appearing

- Check workflow logs in Actions tab
- Verify the `GITHUB_TOKEN` has `pull-requests: write` permission
- Ensure the PR is not from a fork (requires special handling)

### API Rate Limits or Costs

- Set up usage alerts in your AI provider console
- Consider adding rate limiting to the workflow
- Restrict workflow to specific branches or file types

## Cost Estimation

### Claude (Anthropic)
- Input: ~$3 per million tokens
- Output: ~$15 per million tokens
- Typical PR review: $0.01 - $0.10

### OpenAI GPT-4
- Input: ~$10 per million tokens
- Output: ~$30 per million tokens
- Typical PR review: $0.05 - $0.50

### Qwen
- Varies by model and region
- Generally more cost-effective for Chinese regions
- Check Alibaba Cloud pricing

## Security Considerations

1. **Never commit API keys** to the repository
2. Use GitHub Secrets for all sensitive data
3. Review AI suggestions before applying them
4. Consider setting up approval workflows for AI-suggested changes
5. Monitor API usage for unexpected activity

## Support and Contribution

For issues or feature requests, please open a GitHub issue.

Contributions welcome:
- Additional AI provider integrations
- Enhanced review prompts
- Configuration improvements
- Documentation updates
