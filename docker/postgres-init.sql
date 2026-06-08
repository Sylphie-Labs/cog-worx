-- Enable the two extensions cog-worx's single Postgres cluster hosts (CANON S3):
--   timescaledb — hypertable-fast storage for the durable run journal (S6)
--   vector      — pgvector for the hot/cold latent space
-- Tables/hypertables are created idempotently by the adapters (TimescaleJournal, PgLatentStore) so
-- ephemeral test instances can drop+recreate per case; this file only ensures the extensions exist.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
