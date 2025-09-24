### Part C – Short Answer (Reasoning)

1.  If you only had 200 labeled replies, how would you improve the model without collecting thousands more?
    With a small dataset, **data augmentation** is crucial. You could use an LLM to generate synthetic examples by paraphrasing the existing replies, or by creating new ones with similar intent. Additionally, employing **transfer learning** with a pre-trained model like a small transformer would provide a strong starting point and prevent overfitting on the limited data.

2.  How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?
    To prevent bias, I would perform a **fairness analysis** by evaluating the model on different demographic or language subsets, if applicable, to identify and mitigate performance disparities. For safety, I would implement a **multi-layered approach** that includes an external "safety layer" to filter out or flag potentially unsafe outputs before they are sent to the sales reps. Continuous monitoring for **data and concept drift** is also essential.

3.  Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?
    I would use a **persona-based prompt** to set the context, like "You are a senior sales representative..." I'd also incorporate **few-shot examples** to demonstrate the desired style and tone, and use **retrieval-augmented generation (RAG)** by providing specific, up-to-date information about the prospect's company or recent activities to the LLM as context.
