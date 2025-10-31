# Required Repository Secrets

To enable AI-powered PR reviews, you need to add API keys as GitHub repository secrets.

## How to Add Secrets

1. Go to your repository on GitHub
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add the appropriate secret(s) from the options below

---

## Option 1: Grok (xAI) - FREE & RECOMMENDED ✅ 💚

**Current default configuration - 100% FREE!**

### Secret Name
```
XAI_API_KEY
```

### Get Your FREE API Key
1. Visit: https://console.x.ai/
2. Sign up (no credit card required!)
3. Navigate to API keys section
4. Create a new API key
5. Copy the key

### Pricing
- **100% FREE** with generous free tier! 🎉
- No credit card required
- Perfect for open source projects
- Typical PR review: $0.00 (FREE!)

---

## Option 2: Claude (Anthropic)

**Requires workflow modification - PAID**

### Secret Name
```
ANTHROPIC_API_KEY
```

### Get Your API Key
1. Visit: https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Copy the key

### Additional Setup Required
1. Edit `.github/workflows/ai-pr-review.yml`
2. Change install dependencies to: `pip install anthropic requests`
3. Update environment variable to use `ANTHROPIC_API_KEY`
4. Update the Python script to use Anthropic client

### Pricing
- Pay-as-you-go: ~$3 per million input tokens
- Typical PR review: $0.01 - $0.10

---

## Option 3: OpenAI GPT

**Requires workflow modification - PAID**

### Secret Name
```
OPENAI_API_KEY
```

### Get Your API Key
1. Visit: https://platform.openai.com/
2. Sign up or log in
3. Navigate to API keys
4. Create a new secret key
5. Copy the key

### Additional Setup Required
1. Edit `.github/workflows/ai-pr-review.yml`
2. Change install dependencies to: `pip install openai requests`
3. Update Python script to use `multi_provider_reviewer.py`

### Pricing
- GPT-4 Turbo: ~$10 per million input tokens
- Typical PR review: $0.05 - $0.50

---

## Option 3: Qwen (Alibaba Cloud)

**Requires workflow modification and implementation**

### Secret Name
```
QWEN_API_KEY
```

### Get Your API Key
1. Visit: https://dashscope.aliyun.com/
2. Sign up for Alibaba Cloud account
3. Enable DashScope service
4. Get your API key

### Additional Setup Required
1. Install Qwen SDK in workflow: `pip install dashscope`
2. Implement Qwen integration in `multi_provider_reviewer.py`
3. Update workflow to use multi-provider script

### Pricing
- Varies by region and model
- Generally more cost-effective for Chinese regions

---

## Verify Your Setup

After adding the secret(s):

1. Create a test branch: `git checkout -b test-ai-review`
2. Make a small code change
3. Commit and push: `git commit -am "test" && git push`
4. Open a pull request
5. Check the **Actions** tab to see the workflow running
6. Within 1-2 minutes, you should see an AI review comment on your PR

---

## Security Best Practices

✅ **DO:**
- Store API keys only in GitHub Secrets
- Rotate keys periodically (e.g., every 90 days)
- Set up usage alerts in your API provider console
- Use separate API keys for different projects
- Monitor API usage regularly

❌ **DON'T:**
- Never commit API keys to the repository
- Don't share API keys in chat or email
- Don't use the same key across multiple services
- Don't ignore unusual usage patterns

---

## Troubleshooting

### Secret not working?
- Verify the secret name is exactly as shown (case-sensitive)
- Check there are no extra spaces or characters
- Make sure you copied the full API key
- Verify the secret is in the correct repository

### Need to update a secret?
1. Go to Settings → Secrets and variables → Actions
2. Click on the secret name
3. Click "Update secret"
4. Enter the new value
5. Click "Update secret"

### Multiple providers?
You can add multiple API key secrets. The `multi_provider_reviewer.py` script will automatically use the first available one in this order:
1. ANTHROPIC_API_KEY (Claude)
2. OPENAI_API_KEY (OpenAI)
3. QWEN_API_KEY (Qwen)

---

## Cost Management

To control costs:

1. **Set usage limits** in your API provider console
2. **Set up alerts** to notify you of high usage
3. **Exclude files** by editing `.github/ai-review-config.yml`:
   ```yaml
   EXCLUDE_PATTERNS:
     - "*.md"
     - "docs/**"
     - "test/**"
   ```
4. **Reduce verbosity** by setting `MAX_TOKENS: 2048` in config
5. **Monitor usage** regularly in your API dashboard

---

## Need Help?

- 📖 [Setup Guide](AI_REVIEW_SETUP.md)
- ❓ [FAQ](FAQ.md)
- 📋 [Quick Reference](QUICK_REFERENCE.md)
- 💡 [Example Review](EXAMPLE_REVIEW.md)

Or open an issue on GitHub if you encounter problems!
