# What did you learn from implementing a multi-agent workflow： 
1. First It is needed to format their input and output so that they can understand each othe's work and make a perfect work handover.
2. Second, multi-agent means you let different agent to be an expert in different area (you can achieve this by general model difference, fine-tuning, prompts.....)

# Challenge Faced and how to fix:
1. It is hard to have a stable output by description only. I give it a format and write an example in the prompt. 
2. How to internally cite the information. I use the <section_name> content </section_name> as an container. If needed, you can just cite " Based on <section_name>, ...". It is also useful in chain of thought.
3. Model is hard to understand a complex task. I split it into many simple tasks.
4. It is hard to control the font. I telled LLM to use markdown and make some limitation order, it works at most of time, but still have one or two places of mixed font style. I can't fix it... although I tried many methods on prompts.

# Any creative ideas, variations, or design choices 
1. One/Few Shot Prompt
2. Chain of Thoughts
3. Giving persona
4. Structured prompt



