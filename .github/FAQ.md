# AI Code Review - Frequently Asked Questions

## General Questions

### Q: How does the AI review work?
**A:** When you open or update a PR, a GitHub Actions workflow automatically:
1. Fetches the diff of your changes
2. Sends it to the AI model (Claude, GPT, or Qwen)
3. Receives detailed feedback
4. Posts the review as a comment on your PR

### Q: Which AI models are supported?
**A:** Currently supported:
- **Claude 3.5 Sonnet** (Anthropic) - ✅ Primary implementation
- **OpenAI GPT-4** - 🔧 Extension ready (requires integration)
- **Qwen** (Alibaba) - 🔧 Extension ready (requires integration)

You can easily add other AI providers by modifying the reviewer scripts.

### Q: Does this replace human code review?
**A:** No! AI review is a **complement** to human review, not a replacement. It helps catch common issues quickly, but human expertise is essential for:
- Understanding business context
- Evaluating architectural decisions
- Assessing project-specific requirements
- Final approval decisions

---

## Setup and Configuration

### Q: How do I set up AI reviews?
**A:** Three simple steps:
1. Get an API key from your chosen provider (e.g., Anthropic)
2. Add it as a GitHub repository secret named `ANTHROPIC_API_KEY`
3. Open a PR to test - the review will automatically run!

See [AI_REVIEW_SETUP.md](AI_REVIEW_SETUP.md) for detailed instructions.

### Q: Can I use multiple AI providers?
**A:** Yes! The `multi_provider_reviewer.py` script supports multiple providers. It automatically uses the first available API key (Claude → OpenAI → Qwen).

### Q: How do I customize what gets reviewed?
**A:** Edit `.github/ai-review-config.yml`:
```yaml
EXCLUDE_PATTERNS:
  - "*.md"          # Skip markdown files
  - "docs/**"       # Skip documentation
  - "test/fixtures/**"  # Skip test data
```

---

## Cost and Usage

### Q: How much does it cost?
**A:** Costs vary by provider:
- **Claude**: ~$0.01-0.10 per typical PR review
- **OpenAI GPT-4**: ~$0.05-0.50 per review
- **Qwen**: Varies by region, generally cheaper in Asia

Actual costs depend on the size of your PRs.

### Q: How can I control costs?
**A:**
1. **Exclude files** - Don't review docs, tests, or generated files
2. **Limit tokens** - Set `MAX_TOKENS` lower in config
3. **Monitor usage** - Set up alerts in your API provider console
4. **Selective triggers** - Modify workflow to only run on specific branches/labels

### Q: What happens if I hit rate limits?
**A:** The workflow will fail with an error message. You can:
- Wait for the rate limit to reset
- Upgrade your API plan
- Implement retry logic in the script

---

## Technical Questions

### Q: Why isn't the review appearing on my PR?
**A:** Common causes:
1. **Workflow failed** - Check the Actions tab for errors
2. **No API key** - Verify `ANTHROPIC_API_KEY` is set in secrets
3. **Permissions** - Ensure `GITHUB_TOKEN` has write access to PRs
4. **Fork PRs** - Reviews may not work on PRs from forks (security restriction)

### Q: Can I run the review locally?
**A:** Yes! 
```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
export PR_NUMBER="123"
export REPO_NAME="owner/repo"

# Create mock diff files
git diff > pr_diff.txt
git diff --name-status > changed_files.txt

# Run the reviewer
python .github/workflows/ai_reviewer.py
```

### Q: How do I debug workflow failures?
**A:**
1. Go to **Actions** tab in GitHub
2. Click on the failed workflow run
3. Expand the failed step to see error logs
4. Common issues:
   - Missing API key
   - Invalid Python syntax
   - Network/API errors
   - Missing dependencies

### Q: Can I review PRs from forks?
**A:** This requires special configuration because GitHub restricts secrets access from forks for security. Options:
1. Use `pull_request_target` trigger (be careful - security implications)
2. Require contributors to open PRs from branches, not forks
3. Run reviews only after manual approval

---

## Customization

### Q: How do I change the AI model?
**A:** Edit the model parameter in the reviewer script:
```python
# For Claude
model="claude-3-5-sonnet-20241022"  # Current
model="claude-3-opus-20240229"      # More capable, more expensive

# For OpenAI
model="gpt-4-turbo"     # Fast and capable
model="gpt-4"           # Most capable
```

### Q: Can I customize the review prompts?
**A:** Yes! Edit the `prompt` variable in `ai_reviewer.py` or `multi_provider_reviewer.py`. You can:
- Focus on specific aspects (security, performance, etc.)
- Add project-specific guidelines
- Change the tone or format
- Request specific output format

### Q: How do I add a new AI provider?
**A:**
1. Install the provider's SDK
2. Add a new function like `review_with_<provider>()` 
3. Update the provider detection logic
4. Add the API key to repository secrets
5. Update documentation

---

## Best Practices

### Q: What should I do with AI feedback?
**A:**
1. **Read carefully** - Understand the suggestion
2. **Evaluate** - Does it apply to your situation?
3. **Research** - Look up unfamiliar concepts
4. **Discuss** - Ask human reviewers if unsure
5. **Implement** - Apply valid suggestions
6. **Learn** - Use feedback to improve your skills

### Q: Should I trust all AI suggestions?
**A:** No! AI can make mistakes or miss context. Always:
- Verify suggestions match your requirements
- Test changes thoroughly
- Get human review for critical changes
- Use your judgment and expertise

### Q: How often should the AI review run?
**A:** Current setup runs on:
- PR opened
- PR synchronized (new commits)
- PR reopened

This is usually sufficient. Running on every commit might be too frequent and costly.

---

## Security

### Q: Is it safe to send my code to AI providers?
**A:** Consider:
- **Public repos**: Generally fine
- **Private repos**: Review provider's terms of service
- **Sensitive code**: May want to exclude certain files
- **Compliance**: Check if your organization allows it

Most major AI providers (Anthropic, OpenAI) don't use API data for training by default.

### Q: How do I keep API keys secure?
**A:**
- ✅ Store in GitHub Secrets, never in code
- ✅ Rotate keys regularly
- ✅ Monitor usage for anomalies
- ✅ Use separate keys per project
- ✅ Set up usage alerts
- ❌ Never commit keys to repository
- ❌ Don't share keys in team chat

---

## Troubleshooting

### Q: Error: "anthropic module not found"
**A:** The workflow installs dependencies automatically. If you see this error:
- Check the "Install dependencies" step in workflow logs
- Verify pip install command is correct
- Check for Python version compatibility

### Q: Review is very generic/unhelpful
**A:** Try:
- Increasing `MAX_TOKENS` for longer responses
- Customizing the prompt to be more specific
- Using a more capable model (e.g., Claude Opus)
- Ensuring your changes are substantial enough to review

### Q: Workflow times out
**A:**
- Large PRs may take longer to review
- Increase timeout in workflow file
- Consider splitting large PRs into smaller ones
- Implement pagination for very large diffs

---

## Getting Help

### Q: Where can I get more help?
**A:**
- Check [AI_REVIEW_SETUP.md](AI_REVIEW_SETUP.md) for setup details
- Review [EXAMPLE_REVIEW.md](EXAMPLE_REVIEW.md) for sample output
- Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for quick tips
- Open a GitHub issue for bugs or feature requests
- Check Actions logs for technical errors

### Q: How can I contribute?
**A:** Contributions welcome!
- Add new AI provider integrations
- Improve review prompts
- Enhance error handling
- Add tests
- Improve documentation
- Share your configuration examples

---

**Still have questions?** Open an issue on GitHub!
