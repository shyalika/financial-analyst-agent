import glob
import os
import sys
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

if not os.getenv("OPENAI_API_KEY"):
    print("❌ Fatal Security Exception: OPENAI_API_KEY configuration variable missing.")
    sys.exit(1)

from src.ingestion.pipeline import StructuralIngestionPipeline
from src.ingestion.vector_store import SQLiteVectorStore

if __name__ == "__main__":
    raw_documents_dir = os.path.join(ROOT_DIR, "data/raw_documents")
    all_pdfs = sorted(glob.glob(os.path.join(raw_documents_dir, "*.pdf")))
    already_ingested = SQLiteVectorStore().get_ingested_source_files()

    to_ingest = [p for p in all_pdfs if os.path.basename(p) not in already_ingested]
    skipped = [p for p in all_pdfs if os.path.basename(p) in already_ingested]

    print(f"Found {len(all_pdfs)} PDFs in {raw_documents_dir}.")
    for path in skipped:
        print(f" - Skipping (already ingested): {os.path.basename(path)}")

    succeeded, failed = [], []
    total_cost = 0.0

    for path in to_ingest:
        # Fresh pipeline per file -> fresh $0.50 (default) FinOps budget per file,
        # so one large document hitting its cap doesn't block the others, and a
        # per-file failure here is caught and reported rather than aborting the
        # whole batch (same resilience principle as src/generation/faithfulness_eval.py).
        pipeline = StructuralIngestionPipeline()
        try:
            pipeline.run_file_ingestion(path)
            succeeded.append((path, pipeline.finops.usage.total_cost_usd))
        except Exception as e:
            print(f"\n💥 Lifecycle Exception Triggered for {os.path.basename(path)}: {str(e)}")
            failed.append((path, str(e)))
        total_cost += pipeline.finops.usage.total_cost_usd

    print("\n=== Ingestion Run Summary ===")
    print(f"Already ingested (skipped): {len(skipped)}")
    print(f"Newly ingested: {len(succeeded)}")
    for path, cost in succeeded:
        print(f" ✅ {os.path.basename(path)} (${cost:.6f})")
    print(f"Failed / budget-capped (possibly partial): {len(failed)}")
    for path, err in failed:
        print(f" ⚠️  {os.path.basename(path)}: {err}")
    print(f"\nTotal Run Costs: ${total_cost:.6f} USD")
