# AI-Stem-Separation
AI Stem Separation tool built using PyTorch and U-Net for our Senior Design Project.

## 🤖 AI-Powered Code Reviews

This repository is equipped with automated AI code reviews powered by **Claude** (Anthropic). When you open a pull request, the AI will automatically review your code and provide feedback.

### Features

- **Automated PR Reviews**: AI analyzes code changes and provides detailed feedback
- **Multiple AI Models**: Currently using Claude 3.5 Sonnet (easily extensible to other models)
- **Comprehensive Analysis**: Reviews cover code quality, security, performance, and best practices
- **Actionable Feedback**: Get specific suggestions for improvement

### Setup Instructions

#### 1. Enable AI Reviews

To enable AI-powered reviews on your PRs, you need to add an Anthropic API key to your repository secrets:

1. Get an API key from [Anthropic Console](https://console.anthropic.com/)
2. Go to your repository's Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `ANTHROPIC_API_KEY`
5. Value: Your Anthropic API key
6. Click "Add secret"

#### 2. Alternative AI Models

The system is designed to be extensible. You can modify `.github/workflows/ai_reviewer.py` to use:

- **OpenAI GPT Models**: Add OpenAI API integration
- **Qwen Models**: Add Alibaba Cloud/Qwen API integration
- **Other AI Services**: Implement your preferred AI service

Configuration can be adjusted in `.github/ai-review-config.yml`.

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
- Claude API pricing: See [Anthropic Pricing](https://www.anthropic.com/pricing)
- Consider setting up usage limits in your API account
- The workflow only runs on PR events to minimize costs
