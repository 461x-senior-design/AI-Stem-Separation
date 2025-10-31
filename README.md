# AI-Stem-Separation

AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project.

[![AI Code Review](https://img.shields.io/badge/AI-Code%20Review%20Enabled-blue?logo=anthropic)](https://www.anthropic.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🤖 AI-Powered Code Reviews

This repository is equipped with **automated AI code reviews** powered by **Claude** (Anthropic). When you open a pull request, the AI will automatically review your code and provide feedback.

> **Quick Start:** [Setup Guide](.github/SECRETS_TEMPLATE.md) | [FAQ](.github/FAQ.md) | [Example Review](.github/EXAMPLE_REVIEW.md)

### Features

- **Automated PR Reviews**: AI analyzes code changes and provides detailed feedback
- **Multiple AI Models**: Currently using Claude 3.5 Sonnet (easily extensible to other models)
- **Comprehensive Analysis**: Reviews cover code quality, security, performance, and best practices
- **Actionable Feedback**: Get specific suggestions for improvement

### Setup Instructions

#### 🚀 Quick Setup (3 steps)

1. **Get API Key**: Visit [Anthropic Console](https://console.anthropic.com/)
2. **Add Secret**: Settings → Secrets → Actions → New secret
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your API key
3. **Test**: Open a PR and get AI feedback!

📖 **Detailed instructions:** [Setup Guide](.github/AI_REVIEW_SETUP.md) | [Secrets Template](.github/SECRETS_TEMPLATE.md)

#### 🔧 Alternative AI Models

The system supports multiple AI providers:

| Provider | Status | Setup |
|----------|--------|-------|
| **Claude** (Anthropic) | ✅ Active | Default - just add API key |
| **GPT-4** (OpenAI) | 🔧 Ready | [Instructions](.github/FAQ.md#q-can-i-use-multiple-ai-providers) |
| **Qwen** (Alibaba) | 🔧 Ready | [Instructions](.github/FAQ.md#q-how-do-i-add-a-new-ai-provider) |

See [Multi-Provider Guide](.github/AI_REVIEW_SETUP.md) for switching between models.

### How It Works

1. When a PR is opened or updated, the workflow automatically triggers
2. The workflow fetches the PR diff and changed files
3. Claude AI analyzes the changes
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
pip install anthropic requests

# Set environment variables
export ANTHROPIC_API_KEY="your-key-here"
export GITHUB_TOKEN="your-github-token"
export PR_NUMBER="123"
export REPO_NAME="owner/repo"

# Run the reviewer
python .github/workflows/ai_reviewer.py
```

### Cost Considerations

- AI API calls incur costs based on usage
- Claude API pricing: ~$0.01-0.10 per typical PR review
- See [Anthropic Pricing](https://www.anthropic.com/pricing) for details
- Consider setting up usage limits in your API account
- The workflow only runs on PR events to minimize costs

### 📚 Documentation

- **[Quick Reference](.github/QUICK_REFERENCE.md)** - Fast overview and common tasks
- **[Setup Guide](.github/AI_REVIEW_SETUP.md)** - Detailed setup instructions
- **[FAQ](.github/FAQ.md)** - Common questions and troubleshooting
- **[Example Review](.github/EXAMPLE_REVIEW.md)** - See what AI reviews look like
- **[Secrets Template](.github/SECRETS_TEMPLATE.md)** - API key setup guide

---

## About This Project

This is an AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project. The AI code review system helps maintain code quality across all contributions.
