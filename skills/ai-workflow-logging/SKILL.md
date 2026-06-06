---
name: ai-workflow-logging
description: Use when documenting, auditing, or preparing evidence for how Group 9 used Agent AI/Copilot in the traffic scene understanding project. Covers knowledge base, prompt sequence, discussion logs, decisions, AI boundaries, and teacher-facing explanation.
---

# AI Workflow Logging

## Goal

Record how AI was used as a learning and planning assistant, not as a blind project generator.

Never copy or summarize the current chat/session into `logdiscusssion.md` unless the user explicitly allows it or provides the exact content to log. If permission is not given, keep only the empty template and logging rules.

## Log format

For `logdiscusssion.md`, record entries with:

```text
Date/time:
Context:
User requirement:
Knowledge base given to AI:
Prompt summary:
AI response summary:
Decision made by team:
What was changed:
Evidence/output:
```

Keep the wording natural and concise.

## Required AI sequence

When preparing teacher-facing evidence, follow this order:

1. User's requirement
2. Features
3. Tech solutions
4. Logic + AI
5. Implement
6. Test

## Knowledge base expectation

The AI should receive project context before deeper prompts:

- Project topic.
- Input and expected outputs.
- Dataset choices.
- Model choices.
- Scope and out-of-scope.
- Current repo/docs state.
- What the team wants AI to help with.

## Good AI usage statement

Use this summary when needed:

```text
Nhóm em không yêu cầu AI làm toàn bộ dự án. Trước khi hỏi AI, nhóm em chuẩn bị knowledge base gồm yêu cầu đề tài, input/output, dataset, model, scope và giới hạn. AI được dùng để kiểm tra cách hiểu, góp ý feature, so sánh tech solution, rà logic, gợi ý edge cases và lập checklist test. Phần chốt phạm vi, viết tài liệu, chạy demo và đánh giá kết quả là do nhóm tự thực hiện.
```

## Avoid

Do not make the log look like:

- Copy full assignment into AI and ask it to solve everything.
- Paste only final code with no reasoning.
- Hide the team's decisions.
- Overstate AI as an author of the whole project.
