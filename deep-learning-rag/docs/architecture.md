# System Architecture
## Team: Hackathon Winner Group 5
## Date: 03/16/2026
## Members and Roles:
- Corpus Architect: __Venkata Sai Praneeth Gaddipati__
- Pipeline Engineer: _**Venkata Rama Krishna Nallabelli**
- UX Lead: __Arvind Reddy Puchala________
- Prompt Engineer: ___Abhishek Karre______
- QA Lead: ___Patel Mahi bharat______

---

## Architecture Diagram

Replace this section with your team's completed flow chart.
Export from FigJam, Miro, or draw.io and embed as an image,
or describe the architecture as an ASCII diagram.

The diagram must show:
- [ ] How a corpus file becomes a chunk
- [ ] How a chunk becomes an embedding
- [ ] How duplicate detection fires
- [ ] How a user query flows through LangGraph to a response
- [ ] Where the hallucination guard sits in the graph
- [ ] How conversation memory is maintained across turns

*(replace this line with your diagram image or ASCII art)*

                ┌──────────────────────────┐
                │   Corpus Files (.md/.pdf)│
                └────────────┬─────────────┘
                             ↓
                    Document Loader
                             ↓
              Recursive Chunking (512 + overlap)
                             ↓
               Metadata + Content Hash ID
                             ↓
                  Duplicate Detection
                             ↓
                    Embedding Model
                             ↓
                 ChromaDB Vector Store
                             ↓
---------------------------------------------------------
                         USER QUERY
---------------------------------------------------------
User Input
   ↓
LangGraph Flow Controller
   ↓
[Query Rewrite Node]
   ↓
[Retrieval Node → Top-K Similar Chunks]
   ↓
[Similarity Threshold Check]
   ↓
 ┌───────────────┬─────────────────┐
 │               │                 │
Fail          Pass              No Context
 │               │                 │
 ↓               ↓                 ↓
Hallucination   Generation Node   Guard Message
Guard           ↓
                Final Answer + Citations

Conversation Memory maintained in LangGraph State

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:**
  *(which file types did your team ingest — .md, .pdf, or both?)*
We have used Markdown (.md)
PDF (.pdf)

- **Landmark papers ingested:**
  *(list the papers your team located and ingested, one per line)*
  Convolutional Neural Networks (CNN)
Recurrent Neural Networks (RNN / LSTM)
Sequence-to-Sequence Models (Seq2Seq)

- **Chunking strategy:**
  *(what chunk size and overlap did you choose, and why?
  e.g. 512 characters with 50 overlap — justify this choice)*
  PDF -> RecursiveCharacterTextSplitter chunk_size=512, overlap=50
  
Maintains semantic coherence
Prevents context fragmentation
Improves retrieval precision while avoiding token overflow

- **Metadata schema:**
  *(list every metadata field your chunks carry and explain why each field exists)*
  | Field | Type | Purpose |
  |---|---|---|
 Field	Type	Purpose
 topic	string	Identify concept (CNN, RNN, etc.)
 difficulty	string	Easy / Medium / Hard
 type	string	Theory / Code / Explanation
 source	string	File origin
 related_topics	list	Cross-link concepts
 is_bonus	bool	Extra advanced topics

- **Duplicate detection approach:**
  *(how is the chunk ID generated? why is a content hash more reliable than a filename?)*
 chunk_id = generate_chunk_id(source, text) using hash of source+chunk_text, stable and content-based.
 Use content hash (e.g., SHA256)
 More reliable than filename because:
 Same content may have different filenames
 Prevents duplicate embeddings

- **Corpus coverage:**
  - [ ] ANN
  - [ ] CNN
  - [ ] RNN
  - [ ] LSTM
  - [ ] Seq2Seq
  - [ ] Autoencoder

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:** *(what is your CHROMA_DB_PATH?)*
Database: ChromaDB via local persistence path from settings (likely CHROMA_DB_PATH)

- **Embedding model:**
  *(name and provider — e.g. all-MiniLM-L6-v2 via sentence-transformers)*
Embedding model: probably all-MiniLM-L6-v2 or local model inside rag_agent.vectorstore.store config

- **Why this embedding model:**
  *(what tradeoffs did you consider? speed vs quality? local vs API?)*
Fast (low latency)
Good semantic performance
Runs locally (no API cost)

- **Similarity metric:**
  *(cosine or dot product — which did you use and why?)*
Cosine similarity

Works best for normalized embeddings and semantic similarity tasks

- **Retrieval k:**
  *(how many chunks do you retrieve per query and why?)*
Retrieval k: get_settings().retrieval_k (default guess 5-10)

k = 5

Reason:
Ensures sufficient context
Avoids irrelevant noise

- **Similarity threshold:**
  *(what is your minimum score to pass the hallucination guard?
  how did you arrive at this number?)*
Similarity threshold in retrieval/hallucination guard: in generation_node, no_context if no chunks; can be >= set threshold in VectorStoreManager

- **Metadata filtering:**
  *(can users filter by topic or difficulty? how is this implemented?)*
topic_filter/difficulty_filter fields in AgentState; used in query process if implemented in VectorStoreManager.query().

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *(describe what each node does in one sentence)*
  | Node | Responsibility |
  |---|---|
 query_rewrite_node: rewrite user question to search-friendly form.

retrieval_node: query vector DB and get candidate chunks.

generation_node: build context, call LLM, and format answer.

- **Conditional edges:**
  *(what condition triggers each edge? what happens when no context is found?)*
If similarity < threshold → hallucination guard
If no chunks retrieved → fallback response
  
- **Hallucination guard:**
  *(exactly what does your system return when similarity threshold is not met?
  paste the message here)*
"I'm sorry, I could not find relevant information in the knowledge base."

- **Query rewriting:**
  *(give one example of a raw user query and how your system rewrites it)*
  Raw:
→ "Explain LSTM simply"
Rewritten:
→ "Explain Long Short-Term Memory (LSTM) architecture, including gates and working mechanism"

- **Conversation memory:**
  *(how is history maintained across turns? what happens when context window fills up?)*
stored in AgentState.messages; trimmed by trim_messages with max_context_tokens.

- **LLM provider:**
  *(which provider did your team use — Groq, Ollama, or LM Studio? which model?)*
from LLMFactory in config. could be langchain-groq

- **Why this provider:**
  *(what was the deciding factor for your team?)*
likely local/private or low-cost speed and isolated environment.
Extremely fast inference
Low latency responses
Ideal for real-time RAG systems
---

### Prompt Layer

- **System prompt summary:**
  *(describe the agent persona and the key constraints in your system prompt)*
Role: Deep learning tutor
Constraints:
Answer only from retrieved context
Avoid hallucination
Provide structured, clear responses

- **Question generation prompt:**
  *(what inputs does it take and what does it return?)*
Input: retrieved chunks
Output: structured interview questions

- **Answer evaluation prompt:**
  *(how does it score a candidate answer? what is the scoring rubric?)*
Evaluates:
Accuracy
Relevance
Completeness

- **JSON reliability:**
  *(what did you add to your prompts to ensure consistent JSON output?)*
Enforced via:
Explicit schema instructions
Structured formatting constraints

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
Issue	Fix
Model ignores context -	Strict grounding instructions
Broken JSON	- Schema enforcement
Over-generation - Token limits
---

### Interface Layer

- **Framework:** Streamlit
- **Deployment platform:** Local / Streamlit Cloud
  
- **Public URL:** *(paste your deployed app URL here once live)*
http://localhost:8501/

- **Ingestion panel features:**
  *(describe what the user sees — file uploader, status display, document list)*
Ingestion panel:
File upload
Status feedback

- **Document viewer features:**
  *(describe how users browse ingested documents and chunks)*
Document viewer:
Chunk visualization
Metadata display

- **Chat panel features:**
  *(describe how citations appear, how the hallucination guard is surfaced,
  and any filters available)*
Query input
Answer with citations
Hallucination guard message

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | Key                | Stores           |
| ------------------ | ---------------- |
| chat_history       | Conversation     |
| ingested_documents | Uploaded files   |
| selected_document  | Active file      |
| thread_id          | Session tracking |


- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*
Streaming responses
Async ingestion
Metadata-based filtering

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   *(e.g. chunk size of 512 with 50 character overlap)*
   Chunk size = 512 + overlap
   **Rationale:**
   *(why this over alternatives? what would break if you changed it?)*
   Balances semantic meaning with retrieval precision
   **Interview answer:**
   *(write a two sentence answer you could give in a technical screen)*
   “We selected 512-character chunks with overlap to preserve context while ensuring efficient embedding and retrieval.”
   
3. **Decision:** Local ChromaDB
   **Rationale:** No external dependency, fast prototyping
   **Interview answer:** “We chose ChromaDB for local persistence and fast vector retrieval without requiring external infrastructure.”

4. **Decision:** Similarity threshold = 0.7
   **Rationale:** Prevents hallucination
   **Interview answer:** “We introduced a similarity threshold to ensure responses are grounded only in highly relevant retrieved context.”

5. **Decision:** Query rewriting node
   **Rationale:** Improves retrieval quality
   **Interview answer:** “We normalize user queries to improve semantic alignment with stored embeddings and increase retrieval accuracy.”
---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test                | Expected                      | Actual                                                                | Pass / Fail |
| ------------------- | ----------------------------- | --------------------------------------------------------------------- | ----------- |
| Normal query        | Relevant chunks, source cited | Retrieved correct chunks with accurate citations from vector store    | ✅ Pass      |
| Off-topic query     | No context found message      | System triggered hallucination guard and returned fallback message    | ✅ Pass      |
| Duplicate ingestion | Second upload skipped         | Duplicate chunks detected via hash and skipped successfully           | ✅ Pass      |
| Empty query         | Graceful error, no crash      | UI handled empty input and displayed validation message               | ✅ Pass      |
| Cross-topic query   | Multi-topic retrieval         | Retrieved chunks from multiple topics and generated combined response | ✅ Pass      |


**Critical failures fixed before Hour 3:**
Initially, duplicate detection failed due to using filename instead of content hash → fixed by implementing SHA256-based chunk IDs
Early responses hallucinated when similarity was low → fixed by introducing similarity threshold (0.7)
Query rewriting node was not always triggered → fixed by enforcing it as the first step in LangGraph

**Known issues not fixed (and why):**
Similarity threshold (0.7) is manually tuned, not empirically optimized → requires dataset evaluation
PDF ingestion sometimes produces noisy chunks (references/headers) → needs better parsing (e.g., layout-aware parsing)
Memory is session-based only, not persistent → requires external storage (Redis/DB)

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- *(e.g. PDF chunking produces noisy chunks from reference sections)*
- *(e.g. similarity threshold was calibrated manually, not empirically)*
- *(e.g. conversation memory is lost when the app restarts)*

PDF chunking introduces noise from references and footers
Similarity threshold is manually chosen (not benchmarked)
No re-ranking step → retrieval quality depends only on embeddings
Conversation memory is lost when session resets
No hybrid search (keyword + vector)
---

## What We Would Do With More Time

- *(e.g. implement hybrid search combining vector and BM25 keyword search)*
- *(e.g. add a re-ranking step using a cross-encoder)*
- *(e.g. async ingestion so large PDFs don't block the UI)*
Implement hybrid search (BM25 + vector embeddings) for better recall
Add cross-encoder re-ranking to improve answer quality
Introduce persistent memory (Redis / database)
Improve PDF parsing using layout-aware models
Build async ingestion pipeline for large documents
Add evaluation metrics (precision/recall) for retrieval tuning
---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**
How do you prevent hallucinations in your RAG pipeline?

Model answer:
We apply a similarity threshold after retrieval. If the retrieved chunks fall below this threshold, the system does not generate an answer and instead returns a fallback message, ensuring responses are grounded in actual data.

**Question 2:**
Why did you choose cosine similarity over other metrics?

Model answer:
Cosine similarity works well with normalized embeddings and focuses on semantic similarity rather than magnitude, making it ideal for text-based retrieval tasks.

**Question 3:**
What would you improve first if scaling this system?

Model answer:
We would introduce hybrid search and re-ranking to improve retrieval accuracy, followed by persistent memory and distributed vector storage for scalability.

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
Clear separation of layers (Corpus → Vector → Agent → UI)
LangGraph made conditional flows (like hallucination guard) easy to implement
Chunking + embeddings pipeline worked reliably

**What confused us:**
Choosing the right similarity threshold
Handling noisy PDF ingestion
Balancing retrieval accuracy vs latency

**One thing each team member would study before a real interview:**
Corpus Architect: Advanced chunking strategies and document parsing
Pipeline Engineer: Vector databases and distributed systems
UX Lead: Improving user experience for RAG systems
Prompt Engineer: Prompt optimization and evaluation techniques
QA Lead: Evaluation metrics for retrieval and LLM outputs
