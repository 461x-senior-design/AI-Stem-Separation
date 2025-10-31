# AI-Stem-Separation

AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project.

[![AI Code Review](https://img.shields.io/badge/AI-Code%20Review%20Enabled-blue?logo=xai)](https://x.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Free Tier](https://img.shields.io/badge/Cost-FREE-green)](https://console.x.ai/)

## 🤖 AI-Powered Code Reviews (FREE!)

This repository is equipped with **automated AI code reviews** powered by **Grok** (xAI) - completely **FREE** with xAI's generous free tier! When you open a pull request, the AI will automatically review your code and provide feedback.

> **Quick Start:** [Setup Guide](.github/SECRETS_TEMPLATE.md) | [FAQ](.github/FAQ.md) | [Example Review](.github/EXAMPLE_REVIEW.md)

### Features

- **Automated PR Reviews**: AI analyzes code changes and provides detailed feedback
- **100% FREE**: Using xAI's Grok with free tier access - no costs!
- **Fast & Powerful**: Grok-beta model provides quick, thorough reviews
- **Comprehensive Analysis**: Reviews cover code quality, security, performance, and best practices
- **Actionable Feedback**: Get specific suggestions for improvement

### Setup Instructions

#### 🚀 Quick Setup (3 steps)

1. **Get FREE API Key**: Visit [xAI Console](https://console.x.ai/) and sign up
2. **Add Secret**: Settings → Secrets → Actions → New secret
   - Name: `XAI_API_KEY`
   - Value: Your FREE API key from xAI
3. **Test**: Open a PR and get AI feedback - completely free!

📖 **Detailed instructions:** [Setup Guide](.github/AI_REVIEW_SETUP.md) | [Secrets Template](.github/SECRETS_TEMPLATE.md)

#### 🔧 Alternative AI Models

The system supports multiple AI providers:

| Provider | Status | Cost |
|----------|--------|------|
| **Grok** (xAI) | ✅ Active | 💚 FREE |
| **Claude** (Anthropic) | 🔧 Ready | 💰 ~$0.01-0.10/review |
| **GPT-4** (OpenAI) | 🔧 Ready | 💰 ~$0.05-0.50/review |

See [Multi-Provider Guide](.github/AI_REVIEW_SETUP.md) for switching between models.

### How It Works

1. When a PR is opened or updated, the workflow automatically triggers
2. The workflow fetches the PR diff and changed files
3. Grok AI analyzes the changes (using xAI's FREE tier)
4. A detailed review is posted as a comment on the PR
5. Developers can address the feedback before merging

### Configuration

Edit `.github/ai-review-config.yml` to customize:
- AI model selection
- Review focus areas
- Verbosity level
- File exclusion patterns

### Manual Testing

You can test the AI review locally by running:

```bash
# Install dependencies
pip install openai requests

# Set environment variables
export XAI_API_KEY="your-free-key-here"
export GITHUB_TOKEN="your-github-token"
export PR_NUMBER="123"
export REPO_NAME="owner/repo"

# Run the reviewer
python .github/workflows/ai_reviewer.py
```

### Cost Considerations

- **100% FREE** using xAI's Grok free tier! 🎉
- No credit card required for basic usage
- Generous rate limits for open source projects
- Get your FREE API key at [xAI Console](https://console.x.ai/)
- The workflow only runs on PR events to minimize usage

### 📚 Documentation

- **[Quick Reference](.github/QUICK_REFERENCE.md)** - Fast overview and common tasks
- **[Setup Guide](.github/AI_REVIEW_SETUP.md)** - Detailed setup instructions
- **[FAQ](.github/FAQ.md)** - Common questions and troubleshooting
- **[Example Review](.github/EXAMPLE_REVIEW.md)** - See what AI reviews look like
- **[Secrets Template](.github/SECRETS_TEMPLATE.md)** - FREE API key setup guide

---

## About This Project

This is an AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project. The AI code review system helps maintain code quality across all contributions.
