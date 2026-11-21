# Production note: NovaMart support assistant on Google Cloud

## 1. Gemini Enterprise vs. custom build: use both, for different parts
- **Gemini Enterprise as the front door.** It already gives store employees a
  secure chat surface with SSO, enterprise search over connected sources with
  citations, respect for source permissions, and admin controls. Rebuilding
  that would be cost without differentiation. Policy Q&A (most of the volume)
  can run here with almost no custom code.
- **A custom agent for what is specific to NovaMart**: live order lookups and
  later actions (cancel, refund, carrier trace), plus business rules such as the
  date logic for return windows. Build it with **ADK**, deploy on **Vertex AI
  Agent Engine**, and register it in Gemini Enterprise so employees have one
  entry point. This prototype's tools and prompt port directly.
- Decision point for the client: if Gemini Enterprise's out-of-the-box answer
  quality on their documents meets the eval bar, keep policy Q&A there; if not
  (e.g. complex date or eligibility reasoning), route it to the custom agent
  backed by **Vertex AI Search** as the retrieval layer.

## 2. Connecting NovaMart's real data
- **Policies**: keep the documents where policy owners already edit them
  (SharePoint, Google Drive, Confluence or Cloud Storage) and connect them
  through Gemini Enterprise connectors or a Vertex AI Search data store with
  scheduled sync. Require an owner and a "last reviewed" date per document, and
  filter by metadata (region, store type, effective date) so Quebec or holiday
  rules apply correctly.
- **One knowledge base, kept current**: Vertex AI Search builds and maintains
  the vector index itself. The documents stay the source of truth and are
  re-indexed when they change, so answers come from the current version of a
  policy, and the indexed documents become NovaMart's searchable knowledge base
  for other tools too.
- **Orders**: call the order management system through an API layer (e.g.
  Apigee) with read-only scopes first, rate limits and a stable contract, never
  direct database access. Return only the fields the agent needs.

## 3. Security and governance
- **Identity**: SSO with NovaMart's identity provider (Cloud Identity or Workforce
  Identity Federation). Documents are filtered by the user's existing permissions,
  and the agent calls the order API with a least-privilege service account plus
  the user's identity for authorization and audit.
- **Data protection**: VPC Service Controls, CMEK, and a Canadian region where
  data residency requires it; Google does not train its models on enterprise
  data. Redact personal data from logs with Sensitive Data Protection.
- **Model safety**: Model Armor (or equivalent) screening for prompt injection
  and sensitive data; tool output treated as data; any action that changes an
  order needs explicit employee confirmation and is logged.
- **Governance**: Cloud Audit Logs for every tool call, a named business owner
  for content, a change process for prompts and models (versioned, reviewed, gated by
  the eval), and a clear escalation path to the operations team.
- **Personas and access**: Using the roles NovaMart already manages as groups in
  its identity provider (through Workforce Identity Federation), personas and accesses can be managed. A role decides which features and data an employee can use: for example, store staff can look up orders, while only supervisors could approve refunds once actions are added. The check is
  enforced in the backend APIs and tools. In addition to the guarding of backend APIs, hiding buttons in the UI would help improve the user experience.

## 4. Evaluating and monitoring quality over time
- **Before launch**: grow the golden set from real historical tickets with the
  operations team (a few hundred cases, including questions that must be
  declined). Score correctness, groundedness, citation validity, tool-call
  accuracy, and refusal precision and recall, using deterministic checks plus an
  LLM judge (this repo's harness or the Vertex AI Gen AI evaluation service)
  calibrated against human ratings. Run it in CI as a release gate for every
  change to prompts, models, retrieval settings or documents.
- **Rollout**: pilot in a few stores, compare with the existing ticket flow,
  then expand. Pin model versions and canary any upgrade against the eval set.
- **In production**: trace every turn (Cloud Trace and Logging through ADK's
  OpenTelemetry support); dashboards for latency, cost, refusal rate, tool errors and
  user feedback; weekly LLM-judge sampling of live traffic with human review of
  disagreements. Unanswered questions become a backlog for policy owners
  (documentation gaps), and reviewed failures are added to the golden set.
- **Business metric**: reduction in tickets to the operations team and time to
  answer, with no increase in policy errors found in audits.
