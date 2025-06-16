from pipeline import StockMovementPipeline
from datetime import datetime, timedelta

pipeline = StockMovementPipeline()

current_date = datetime.now()
print(current_date)
start_month = (current_date - timedelta(days=21*30)).strftime('%Y-%m')
print(timedelta(days=21*30))
print(current_date-timedelta(days=21*30))
print(start_month)
# pipeline.extract(start_date=start_month + "-01")
# pipeline.transform()
# pipeline.load()

print(f"Successfully processed data for {pipeline.symbol} from {start_month}")
