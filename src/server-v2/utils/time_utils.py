# utils/time_utils.py
from num2words import num2words
from datetime import datetime
import pytz

def time_to_words():
    """Convert current IST time to words (e.g., '4:04' to 'four hours and four minutes', '4:00' to 'four o'clock')."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    hour = now.hour % 12 or 12  # Convert 24-hour to 12-hour format (0 -> 12)
    minute = now.minute
    
    # Convert hour to words
    hour_word = num2words(hour, to='cardinal')
    
    # Handle minutes
    if minute == 0:
        return f"{hour_word} o'clock"
    else:
        minute_word = num2words(minute, to='cardinal')
        return f"{hour_word} hours and {minute_word} minutes"