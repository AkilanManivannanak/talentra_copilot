# Postmortem: what broke and how this version fixed it

## Incident 1: OpenAI quota and auth failures blocked ingestion
- **Symptom**: resume upload returned 500 because embeddings failed.
- **Root cause**: the earlier version depended on OpenAI embeddings and chat completions, and the API project had invalid or exhausted quota.
- **Fix**: replaced the default path with local lexical retrieval and rule-based extraction; external API spend is now optional and disabled by default.

## Incident 2: brittle PDF ingestion caused whole-batch failure
- **Symptom**: one malformed PDF crashed the entire upload request.
- **Root cause**: upload processing was all-or-nothing with no per-file error isolation.
- **Fix**: upload now supports partial success; unreadable files return structured failures instead of raw 500s.

## Incident 3: ghost candidates after failed ingest
- **Symptom**: failed uploads still left candidate records behind.
- **Root cause**: metadata was written before ingest fully succeeded.
- **Fix**: candidate/document cleanup now runs on failure and orphan candidates are removed.

## Incident 4: Copilot contradicted evaluation ranking
- **Symptom**: Evaluate ranked Akila first while Copilot sometimes answered Jaxon or Esha.
- **Root cause**: comparison questions were answered from raw lexical hits instead of the evaluation results.
- **Fix**: ranking/comparison questions now route to evaluation results first; targeted skill questions use evidence search separately.

## Incident 5: header/contact junk polluted evidence
- **Symptom**: citations surfaced email, LinkedIn, and phone blocks.
- **Root cause**: chunks from resume headers were indexed without downweighting.
- **Fix**: low-signal chunk filtering, contact-noise penalties, and evidence deduping were added.
