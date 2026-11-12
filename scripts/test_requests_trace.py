"""
End-to-end trace tester for user requests.
Verifies all stages and outputs log events.
"""

import asyncio
import time
import io
import sys

# Ensure UTF-8 output on Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from backend.app.database.session import init_db, close_db, get_session_factory
from backend.app.services.conversation_service import ConversationService

async def run_all_tests():
    await init_db()
    factory = get_session_factory()
    
    test_messages = [
        ("Test 1: Single feature", "My brakes are worn out."),
        ("Test 2: Dual features", "My brakes are worn out and battery seems weak."),
        ("Test 3: Age feature", "My car is 5 years old."),
    ]
    
    for title, msg in test_messages:
        print(f"\n{'='*70}\n>>> {title}: {msg!r}\n{'='*70}")
        async with factory() as session:
            service = ConversationService(session)
            t0 = time.time()
            try:
                res = await asyncio.wait_for(
                    service.handle_message(user_message=msg, conversation_id=None),
                    timeout=30.0
                )
                elapsed = time.time() - t0
                print(f"\n[COMPLETED in {elapsed:.2f}s]")
                print(f"Assistant Message: {res.get('message')}")
                pred = res.get('prediction')
                if pred:
                    print(f"Prediction: class={pred.get('prediction_class')}, risk={pred.get('risk_level')}, score={pred.get('model_score')}")
                else:
                    print("Prediction: None (need >= 2 features)")
                expl = res.get('explanation')
                print(f"Explanation: available={bool(expl and expl.get('available'))}")
                if expl and expl.get('top_factors'):
                    print(f"Top Factors: {[(f.get('display_name'), f.get('direction')) for f in expl['top_factors']]}")
            except asyncio.TimeoutError:
                print(f"\n[FAILED - TIMEOUT after {time.time()-t0:.2f}s]")
            except Exception as e:
                print(f"\n[FAILED - EXCEPTION: {e}]")
                import traceback; traceback.print_exc()

    await close_db()

if __name__ == "__main__":
    asyncio.run(run_all_tests())
