# Quick Reference: AI PR Reviews

## 🚀 Getting Started in 3 Steps

### 1. Add API Key
Go to: **Settings → Secrets → Actions → New secret**
- Name: `ANTHROPIC_API_KEY`
- Value: Get from https://console.anthropic.com/

### 2. Open a PR
The AI review runs automatically when you:
- Open a new pull request
- Push new commits to existing PR
- Reopen a PR

### 3. Get Review
Check the PR comments for AI feedback within 1-2 minutes.

---

## 📋 What Gets Reviewed

✅ **Code Quality** - Style, structure, conventions  
✅ **Security** - Vulnerabilities, sensitive data  
✅ **Performance** - Optimization opportunities  
✅ **Bugs** - Potential issues and edge cases  
✅ **Best Practices** - Language-specific patterns  
✅ **Maintainability** - Readability, documentation  

---

## 🔄 Switching AI Models

### Current: Claude 3.5 Sonnet (Anthropic)
- Best for: Detailed code analysis
- Cost: ~$0.01-0.10 per review

### Alternative: OpenAI GPT-4
1. Add `OPENAI_API_KEY` secret
2. Edit workflow to use `multi_provider_reviewer.py`
3. Install `openai` package in workflow

### Alternative: Qwen (Alibaba)
1. Add `QWEN_API_KEY` secret  
2. Implement Qwen integration
3. Update workflow dependencies

---

## ⚙️ Configuration

Edit `.github/ai-review-config.yml`:

```yaml
AI_MODEL: "claude-3-5-sonnet-20241022"
VERBOSITY: "medium"  # low | medium | high
EXCLUDE_PATTERNS:
  - "*.md"
  - "test/fixtures/**"
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No review appears | Check Actions tab for errors |
| "API key not found" | Verify secret name is exact |
| Review is too brief | Increase `VERBOSITY` or `MAX_TOKENS` |
| Workflow fails | Check Python dependencies installed |
| High costs | Add file exclusion patterns |

---

## 💡 Tips

- AI reviews are suggestions, use your judgment
- Review the review before applying changes
- Configure exclusions for generated files
- Monitor API usage in provider console
- Set up usage alerts to avoid surprises

---

## 📊 Workflow Status

Check workflow runs: **Actions → AI PR Review**

Review artifacts: Available for 7 days after run
- `pr_diff.txt` - Changes analyzed
- `changed_files.txt` - Files reviewed
- `review_output.txt` - Full review text

---

## 🔐 Security Notes

⚠️ **Never commit API keys to the repository**  
✅ Always use GitHub Secrets  
✅ Review AI suggestions before implementing  
✅ Keep API keys rotated regularly  
✅ Monitor for unusual API activity  

---

For detailed setup instructions, see [AI_REVIEW_SETUP.md](.github/AI_REVIEW_SETUP.md)
