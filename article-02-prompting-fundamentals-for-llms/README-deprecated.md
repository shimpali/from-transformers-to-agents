# Prompting Fundamentals for LLMs: Sources, Context, and Sycophancy

*Article 2 of "From Transformers to Agents", my learning journey through AI, one course at a time.*

Article 1 covered how LLMs work inside. This one is about using them well. Most of my prompts used to be vague one-liners, so I drew on the slides of DeepLearning.AI's [AI Prompting for Everyone](https://www.deeplearning.ai/courses/ai-prompting-for-everyone) and organized what I learned by concept, not by lesson. I then checked it against two independent sources, [Anthropic's prompting guidance](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) and [a research paper on sycophancy](https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models), and I call out where they add to the course or differ from it. The templates below are my own adaptations of these ideas.

![Four decisions behind a good prompt: pick the source, give context, get honest critique, match the tool](images/01-prompting-overview.svg)

## Series roadmap

This is article 2 of an 11-part series that works from DeepLearning.AI courses and independent sources, grouped into four sections:

1. **Foundation (understanding the basics)**: how transformer LLMs work, prompting fundamentals (this article), function-calling and data extraction.
2. **Agentic AI deep dive (core patterns)**: agentic workflows, the four design patterns, reflection, tools and MCP.
3. **Orchestration (connecting it all)**: planning and multi-agent systems, evaluating agents for production, the A2A protocol.
4. **Deployment & implementation (operating at scale)**: prompting multi-agent systems in practice.

## TL;DR (Cheat sheet)

| When you are... | Do this |
|---|---|
| Asking a question | Pick the source first: **pretrained knowledge, web search, or deep research** |
| Asking about health, finance, or law | **Name the source types** you trust |
| Starting a task | Give **relevant context**, and start a **new chat** when the topic changes |
| Brainstorming | Ask for **multiple options**, react, and **iterate** |
| Tackling something hard | Use the **best model**, add context, give a hard task, and **tell it to think** |
| Writing the prompt itself | Be **clear and direct**, give the **reason** behind rules, add **3 to 5 examples**, and put **long documents first, question last** |
| Wanting honest feedback | Ask **neutrally**, use a **rubric** with the total score last, and try a **second model** |
| Writing | **Outline first**, then expand |
| Generating images or video | Expect **slow and costly**, so describe setting, subject, and style, and generate options |
| Building or analyzing | **Start small**, and let code do the math |
| Choosing how to get it done | Match the **tool** to the task: answer, search, run code, or use a file |

---

## 1. Treat the model as a collaborator, not a search box

- A **novice** types a short question, gets one answer, and moves on. That works for "does this restaurant still serve that taco?" and not much else.
- A **power user** hands over a harder task, supplies context, gives the model time to think, and has a way to judge the result.
- The difference is not secret phrases. It is whether you assume the model can do real work, like comparing options or reading a pile of material, instead of treating it as fancier autocomplete.

---

## 2. Where should the answer come from?

Before phrasing anything, decide where the answer should come from. There are three options, and picking the wrong one is behind a lot of disappointing responses.

**Pretrained knowledge**
- This is everything the model absorbed during training: books, forums, encyclopedias, news, papers. It is why a model can answer everyday and surprisingly niche questions with no lookup at all.
- It is **frozen** at the training cutoff. Ask about something that happened last month and the model has no idea, though it may guess confidently anyway.
- It mirrors its training data, so popular topics are handled well, rare ones less reliably, and old typos and myths came along for the ride.

**Web search**
- This is the fix for frozen knowledge, and it earns its keep on three kinds of questions: **recent** events, **location-specific** needs, and **niche** topics the model is unlikely to know well.
- Search can be triggered by the model's own judgment or by you saying "search the web". If you are not sure it will search, just ask.
- Search is a **pipeline**, not a full read of the internet. The model turns your prompt into several queries, scans what comes back, filters out the irrelevant, and summarizes the rest. So the topic, the kind of sources, and the depth you ask for all shape the answer.
- Left alone, it leans toward whatever is popular and easy to find, which might be a forum thread instead of a solid study. For health, finance, or legal questions, **name the source types you trust** in the prompt.
- Results can be stale too. Asking for places to run in a city can surface a closed park, so add "check that it is still open."

**Deep research**
- This is the heavy option for complex questions. The model makes a plan, searches and reads in parallel, evaluates what it found, loops back if it has gaps, and finally writes a cited report.
- It takes minutes rather than seconds, and can pull from many sources.
- This is the first appearance of the word **agentic** in this series: the model decides for itself what to do next. Keep that in mind, because it is where we are heading.

![Flowchart: choosing between pretrained knowledge, web search, and deep research](images/02-choosing-an-information-source.svg)

*Not sure whether the model will search? Ask for it explicitly.*

- A plain search engine still wins when you just want to reach a specific page or see raw data.
- AI wins when the job is comparing and combining several sources, such as pros and cons of a supplement instead of one product page.

---

## 3. Context is the biggest lever

- **Context** is everything the model can see when it responds: system instructions, tool definitions, chat history, uploaded files, and anything it retrieved. Your prompt is only one slice of it.
- The **context window** is how much fits at once, and modern windows are enormous, on the order of hundreds of thousands of words. More is not automatically better, though. **Relevant beats voluminous**, and old unrelated material can leak into a new task, so when the topic changes, start a new chat.
- **Creativity needs context.** Models drift toward the common, high-probability answer. Ask for a workout plan and you get squats. Add your age, fitness level, time, equipment, and movements to avoid, and you get something you can follow. The recipe is **context, multiple options, feedback, iterate.**
- **Hard tasks need the right setup.** Use the **best available model**, give it **enough context**, give it a **genuinely hard task**, and **tell it to think**. "Read everything and think hard before answering" beats a one-line question.
- **Where sources differ: "think step by step".** Anthropic's guidance prefers a general nudge like "think thoroughly" over a hand-written list of steps, since the model's own reasoning often beats the plan you would prescribe. Writing out manual step-by-step reasoning is a fallback for when built-in thinking is off. It also notes that thinking adds latency, so for simple questions you may want the model to answer directly.
- **Know the ceiling.** Model ability is jagged: a model can do something impressive and then stumble on something that looks simpler. Trying more than one model keeps you calibrated about who is good at what.

---

## 4. Models want to please you

- **Sycophancy** is the model's habit of agreeing with you and praising your ideas instead of analyzing them. It comes from training on human feedback, where people tend to reward agreeable answers, so the habit gets reinforced.
- **The research backs this up.** Anthropic's paper "Towards Understanding Sycophancy in Language Models" tested five state-of-the-art assistants across four open-ended tasks, and all five showed it. In the human preference data it analyzed, responses that matched a user's views were more likely to be preferred. Both human raters and preference models sometimes picked a convincing sycophantic answer over a correct one, and optimizing against those models could trade truthfulness for agreement.
- It is worst on subjective work like ideas and writing, where nobody can quickly tell the flattery is wrong.
- The fix starts with how you ask. A leading question invites a yes ("Aren't carbon taxes bad for small businesses?"), while a neutral one invites analysis ("To what extent, if at all, do carbon taxes affect small businesses?"). A fresh chat also helps get an opinion not colored by earlier conversation.
- For feedback, use a **rubric**. Asking "critique my story" with no criteria gets warm, inflated praise. A rubric turns opinion into checkable conditions: categories, points, and specific criteria.
- **Order matters.** Have the model score and justify each category first and add the total last, otherwise it tends to defend an early number.
- **Get a second opinion.** Paste the same work and rubric into a different model. Disagreement between the two is a useful signal about where to look harder.

A critique template adapted from the rubric idea:

```
Critique the attached work using this rubric.

Rubric:
- Criterion A: [points and explicit standard]
- Criterion B: [points and explicit standard]

For each criterion: assign a score, cite evidence, say what works,
say what to improve. Add up the total at the end.
Be critical. Do not assume the work is already good.
```

---

## 5. Writing with AI: outline first

- **AI slop** is text that reads smoothly but says little. Tells include overused words, reflexive lists of three, and "not X but Y" constructions. Fewer abstract nouns and more concrete detail from you is the cure.
- Do not ask for the full article in one go. Go from **research, to several outlines, to one edited outline, to bullets, to section detail, to the draft.**
- The logic is cost. Changing an outline takes seconds and has a big effect, while restructuring a finished draft is like knocking down walls after the house is built.
- When you do edit, work **piece by piece** instead of asking for a rewrite of everything, so you keep what already works.

---

## 6. Beyond text: images, apps, and data

- **Output type matters more than input type.** Models take in text, images, audio, and more with little practical difference, but what they generate changes cost and speed a lot. Text is fast and cheap, while images and video are slower and pricier.
- That cost changes how you prompt. Giving **more context** and using the **best model** stay easy. **Generating multiple options** and **iterating** get harder because every try is expensive, so think harder before each one.
- **Images as input** are great context (a whiteboard, a receipt, handwriting), and you can attach several at once. But models can miss fine details, such as telling apart two visually similar gym machines, so check anything that matters.
- **Images as output** work best when you describe the **setting**, the **subject details**, and the **mood or style**. Style words like cinematic, watercolor, or anime change the whole look.
- Image models are **diffusion** models that generate the whole image at once instead of piece by piece. That is why you cannot stop one early, and why results vary run to run, with known weak spots like hands, garbled text, and inconsistent characters.
- **Apps are prompts too.** Describe the **goal** (what to create), the **input** (what the user provides), and the **output** (what the app does). Start with a simple idea, since a quiz app is much easier to get right than a live multiplayer game.
- **Data analysis** works because models can **write and run code**. Upload the file and ask a question. The model inspects the data, calculates, and builds the chart, so the AI is the analyst and the code is the calculator.
- **Responsibility comes with power.** The same voice tools that fix a podcast can fake a relative's voice in a scam, so think about how outputs could be misused.

---

## 7. The through-line: choosing the right tool

- Web search, running code, reading files, and writing files are all **tools** the model can call. The model reasons about the prompt, decides whether it needs a tool, uses it, and loops back with more context until it can answer.
- The model picks by the shape of the question. Common knowledge comes from pretrained memory, location-specific questions need search, complex synthesis points to deep research, and anything with numbers or charts needs code.
- Desktop-style assistants can also read and write your files on their own. Use a **plan, review, execute** loop, and grant access only to the folders the task needs.
- The question shifts from "can the model answer this?" to "does this need fresh information, exact math, or a file?"

![Flowchart: choosing between web search, running code, file tools, and answering directly](images/03-choosing-a-tool.svg)

*Complex, multi-source questions can combine these steps, which is what deep research automates.*

- This is where prompting stops being a writing skill and becomes a design skill. It leads straight into function calling (article 3) and agents (article 4).

---

## What's next

Article 3 moves from prompting a model to structuring what it does: **function calling and data extraction**, where output becomes something software can act on.

### Sources

- DeepLearning.AI, ["AI Prompting for Everyone"](https://www.deeplearning.ai/courses/ai-prompting-for-everyone) (Andrew Ng)
- Anthropic, ["Prompt engineering overview"](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- Anthropic, ["Towards Understanding Sycophancy in Language Models"](https://www.anthropic.com/research/towards-understanding-sycophancy-in-language-models)
- Article 1, *How Transformer LLMs Work*

---

The prompt template above is an original adaptation of course concepts, not a reproduction of course materials.

<!-- medium:cut -->
## In this folder

| Path | What it is |
|---|---|
| [`images/`](images) | The 3 diagrams above, as standalone SVGs |

This article is prompt-only, so it has no scripts, notebook or Gists.
