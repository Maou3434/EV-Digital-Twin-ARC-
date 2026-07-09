---
name: humanizer
description: Removes signs of AI-generated writing from text, making it sound more natural and human. Detects 24 patterns of AI writing artifacts and transforms them into human-like prose.
license: MIT
---

This skill removes signs of AI-generated writing from text, making it sound more natural and human. Based on Wikipedia's "Signs of AI writing" guide.

## The 24 Patterns to Detect and Transform

### Content Patterns

1. **Significance inflation**: Transform "marking a pivotal moment in the evolution of..." to specific facts like "was established in 1989 to collect regional statistics"

2. **Notability name-dropping**: Replace "cited in NYT, BBC, FT, and The Hindu" with specific attributions like "In a 2024 NYT interview, she argued..."

3. **Superficial -ing analyses**: Remove or expand "symbolizing... reflecting... showcasing..." with actual sources

4. **Promotional language**: Transform "nestled within the breathtaking region" to factual "is a town in the Gonder region"

5. **Vague attributions**: Replace "Experts believe it plays a crucial role" with specific "according to a 2019 survey by..."

6. **Formulaic challenges**: Transform "Despite challenges... continues to thrive" to specific facts about actual challenges

### Language Patterns

7. **AI vocabulary**: Replace "Additionally... testament... landscape... showcasing" with natural alternatives like "also... remain common"

8. **Copula avoidance**: Transform "serves as... features... boasts" to direct "is... has"

9. **Negative parallelisms**: Replace "It's not just X, it's Y" with direct statements

10. **Rule of three**: Transform "innovation, inspiration, and insights" to a natural number of items

11. **Synonym cycling**: Keep the clearest term and repeat it (e.g., use "protagonist" consistently)

12. **False ranges**: Replace "from the Big Bang to dark matter" with direct topic lists

### Style Patterns

13. **Em dash overuse**: Use commas or periods instead of "institutions—not the people—yet this continues—"

14. **Boldface overuse**: Remove bold from terms like "**OKRs**, **KPIs**, **BMC**"

15. **Inline-header lists**: Convert "**Performance:** Performance improved" to prose

16. **Title Case Headings**: Transform "Strategic Negotiations And Partnerships" to sentence case "Strategic negotiations and partnerships"

17. **Emojis**: Remove emojis like "🚀 Launch Phase: 💡 Key Insight:"

18. **Curly quotes**: Use straight quotes `"the project"` instead of curly quotes `"the project"`

### Communication Patterns

19. **Chatbot artifacts**: Remove "I hope this helps! Let me know if..."

20. **Cutoff disclaimers**: Transform "While details are limited in available sources..." to found sources or removal

21. **Sycophantic tone**: Replace "Great question! You're absolutely right!" with direct responses

### Filler and Hedging

22. **Filler phrases**: Transform "In order to", "Due to the fact that" to "To", "Because"

23. **Excessive hedging**: Replace "could potentially possibly" with "may"

24. **Generic conclusions**: Transform "The future looks bright" to specific plans or facts

## Usage

When the user wants to humanize text:
- Invoke this skill to transform AI-generated text into natural human writing
- Apply the 24 patterns to identify and correct AI artifacts
- Ensure the output sounds natural, factual, and human-written

## Example Transformation

**Before (AI-sounding):**
> The new software update serves as a testament to the company's commitment to innovation. Moreover, it provides a seamless, intuitive, and powerful user experience—ensuring that users can accomplish their goals efficiently. It's not just an update, it's a revolution in how we think about productivity. Industry experts believe this will have a lasting impact on the entire sector, highlighting the company's pivotal role in the evolving technological landscape.

**After (Humanized):**
> The software update adds batch processing, keyboard shortcuts, and offline mode. Early feedback from beta testers has been positive, with most reporting faster task completion.
