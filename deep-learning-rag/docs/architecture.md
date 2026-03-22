# System Architecture
## Team: Hackathon Winner Group 5
## Date: 03/16/2026
## Members and Roles:
- Corpus Architect: __Venkata Sai Praneeth Gaddipati__
- Pipeline Engineer: _**Venkata Rama Krishna Nallabelli**
- UX Lead: ________Arvind Reddy Puchala________
- Prompt Engineer: ______Abhishek Karre______
- QA Lead: _______Patel Mahi bharat______

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

Raw file (PDF/MD) -> DocumentChunker (chunk_file, _chunk_pdf/_chunk_markdown) -> DocumentChunk + ChunkMetadata
   ↓
VectorStoreManager.ingest() -> embed chunks -> dedupe by chunk_id (hash(source+text)) -> ChromaDB
   ↓
User query -> LangGraph:
   query_rewrite_node -> retrieval_node -> should_retry_retrieval
      |- no_context_found=True -> hallucination guard (end)
      |- no_context_found=False -> generation_node -> AgentResponse
   ↓
Conversation memory in AgentState.messages, checkpoint via MemorySaver

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- **File formats used:**
  *(which file types did your team ingest — .md, .pdf, or both?)*
We have used .md, .pdf

- **Landmark papers ingested:**
  *(list the papers your team located and ingested, one per line)*
  -lstm.pdf
  -ANN.odf
  -RNN.pdf

- **Chunking strategy:**
  *(what chunk size and overlap did you choose, and why?
  e.g. 512 characters with 50 overlap — justify this choice)*
  PDF -> RecursiveCharacterTextSplitter chunk_size=512, overlap=50
  Markdown -> MarkdownHeaderTextSplitter + recursive split
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
Works best for normalized embeddings
- **Retrieval k:**
  *(how many chunks do you retrieve per query and why?)*
Retrieval k: get_settings().retrieval_k (default guess 5-10)
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
should_retry_retrieval: returns end if no context, generate if has context.
0-context -> hallucination guard path returns safe message.
- **Hallucination guard:**
  *(exactly what does your system return when similarity threshold is not met?
  paste the message here)*
I was unable to find relevant information ... please try ...
- **Query rewriting:**
  *(give one example of a raw user query and how your system rewrites it)*
  - Raw query:
  - Rewritten query:
How do LSTMs remember information?”
rewritten: maybe “LSTM long-term memory cell state gate mechanisms”
- **Conversation memory:**
  *(how is history maintained across turns? what happens when context window fills up?)*
stored in AgentState.messages; trimmed by trim_messages with max_context_tokens.
- **LLM provider:**
  *(which provider did your team use — Groq, Ollama, or LM Studio? which model?)*
from LLMFactory in config. could be langchain-groq, ollama, or local.
- **Why this provider:**
  *(what was the deciding factor for your team?)*
likely local/private or low-cost speed and isolated environment.
---

### Prompt Layer

- **System prompt summary:**
  *(describe the agent persona and the key constraints in your system prompt)*
rag_agent.agent.prompts.SYSTEM_PROMPT — persona as helpful deep learning content researcher + safety/hallucination guard.
- **Question generation prompt:**
  *(what inputs does it take and what does it return?)*

- **Answer evaluation prompt:**
  *(how does it score a candidate answer? what is the scoring rubric?)*

- **JSON reliability:**
  *(what did you add to your prompts to ensure consistent JSON output?)*

- **Failure modes identified:**
  *(list at least one failure mode per prompt and how you addressed it)*
  -
  -
  -

---

### Interface Layer

- **Framework:** *(Streamlit / Gradio)*
- **Deployment platform:** *(Streamlit Community Cloud / HuggingFace Spaces)*
- **Public URL:** *(paste your deployed app URL here once live)*
http://localhost:8501/
- **Ingestion panel features:**
  *(describe what the user sees — file uploader, status display, document list)*

- **Document viewer features:**
  *(describe how users browse ingested documents and chunks)*

- **Chat panel features:**
  *(describe how citations appear, how the hallucination guard is surfaced,
  and any filters available)*

- **Session state keys:**
  *(list the st.session_state keys your app uses and what each stores)*
  | Key | Stores |
  |---|---|
  | chat_history | |
  | ingested_documents | |
  | selected_document | |
  | thread_id | |

- **Stretch features implemented:**
  *(streaming responses, async ingestion, hybrid search, re-ranking, other)*

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   *(e.g. chunk size of 512 with 50 character overlap)*
   **Rationale:**
   *(why this over alternatives? what would break if you changed it?)*
   **Interview answer:**
   *(write a two sentence answer you could give in a technical screen)*

2. **Decision:**
   **Rationale:**
   **Interview answer:**

3. **Decision:**
   **Rationale:**
   **Interview answer:**

4. **Decision:** *(optional — bonus points in Hour 3)*
   **Rationale:**
   **Interview answer:**

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | | |
| Off-topic query | No context found message | | |
| Duplicate ingestion | Second upload skipped | | |
| Empty query | Graceful error, no crash | | |
| Cross-topic query | Multi-topic retrieval | | |

**Critical failures fixed before Hour 3:**
-
-

**Known issues not fixed (and why):**
-
-

---

## Known Limitations

Be honest. Interviewers respect candidates who understand
the boundaries of their own system.

- *(e.g. PDF chunking produces noisy chunks from reference sections)*
- *(e.g. similarity threshold was calibrated manually, not empirically)*
- *(e.g. conversation memory is lost when the app restarts)*

---

## What We Would Do With More Time

- *(e.g. implement hybrid search combining vector and BM25 keyword search)*
- *(e.g. add a re-ranking step using a cross-encoder)*
- *(e.g. async ingestion so large PDFs don't block the UI)*

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:**

Model answer:

**Question 2:**

Model answer:

**Question 3:**

Model answer:

---

## Team Retrospective

*(fill in after Hour 3)*

**What clicked:**
-

**What confused us:**
-

**One thing each team member would study before a real interview:**
- Corpus Architect:
- Pipeline Engineer:
- UX Lead:
- Prompt Engineer:
- QA Lead:
