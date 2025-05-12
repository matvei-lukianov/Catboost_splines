import json
import heapq
from collections import defaultdict
import pandas as pd
import numpy as np

class OrderBook:
    def __init__(self, num_levels=10):
        self.num_levels = num_levels
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)
        self.last_update_id = None
        self.got_snaphot = False

    def update_order_book(self, bids, asks):
        # Update bids
        for price, qty in bids:
            price = float(price)
            qty = float(qty)

            # Ensure the quantity is non-negative
            assert qty >= 0, f"Bid quantity cannot be negative: {qty}"

            if qty == 0:
                self.bids.pop(price, None)  # Remove level if quantity is 0
            else:
                self.bids[price] = qty  # Update or add the bid
        self.got_snaphot = False

        # Update asks
        for price, qty in asks:
            price = float(price)
            qty = float(qty)

            assert qty >= 0, f"Ask quantity cannot be negative: {qty}"

            if qty == 0:
                 self.asks.pop(price, None) 
            else:
                self.asks[price] = qty

        # Check that the lowest ask is greater than the highest bid
        if self.bids and self.asks:
            max_bid = max(self.bids)
            min_ask = min(self.asks)
            assert min_ask > max_bid, f"Ask price ({min_ask}) must be greater than bid price ({max_bid})"

    def process_snapshot(self, bids, asks, last_update_id):
        # Initialize the order book from a snapshot
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)
        self.update_order_book(bids, asks)
        self.last_update_id = last_update_id
        self.got_snaphot = True

    def get_top_levels(self):
        top_bids = heapq.nlargest(self.num_levels, self.bids.items())
        top_asks = heapq.nsmallest(self.num_levels, self.asks.items())
        return top_bids, top_asks


def process_order_book_at_timestamp(file_path, target_timestamp):
    order_book = OrderBook()
    prev_message = {}

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            message = data['m']
            
            
            # Check if it's a snapshot (no 'e' key)
            if 'lastUpdateId' in message:
                last_update_id = message['lastUpdateId']
                bids = message['bids']
                asks = message['asks']
                
                order_book.process_snapshot(bids, asks, last_update_id)

            # Check if it's a depthUpdate event
            elif message['e'] == 'depthUpdate':
                event_time = message['E']
                if event_time > target_timestamp:
                    break  # Stop processing once the target timestamp is exceeded

                # Skip events where u is <= lastUpdateId
                if order_book.last_update_id and message['u'] <= order_book.last_update_id:
                    prev_message = message
                    print('skipped depthUpdate')
                    continue
                
                # Ensure the first processed event after snapshot has U <= lastUpdateId + 1 AND u >= lastUpdateId + 1
                if order_book.got_snaphot and not (message['U'] <= order_book.last_update_id + 1):
                    if (prev_message['U'] <= order_book.last_update_id + 1):
                        order_book.update_order_book(prev_message['b'], prev_message['a'])
                    else:
                        print('broken stream')
                        break
                if 'u' in prev_message.keys():
                    if prev_message['u'] +1 != message['U']:
                        print('broken stream')
                        break

                # Update the order book with new bids and asks
                bids = message['b']
                asks = message['a']
                order_book.update_order_book(bids, asks)
                prev_message = message

    # Get the top 10 levels of bids and asks
    top_bids, top_asks = order_book.get_top_levels()
    return top_bids, top_asks


def create_orderbook_dataframe(file_path, start_timestamp, end_timestamp):
    order_book = OrderBook()
    data_list = []

    current_second = start_timestamp // 1000
    next_second = current_second + 1

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data = data['m']
            
            event_time = data['E']
            if data['e'] == 'depthUpdate':
                if event_time > end_timestamp:
                    break  # Stop processing once the end timestamp is exceeded
                
                order_book.update_order_book(data['b'], data['a'])
                
                # If we've reached the next second, capture the top 10 bids and asks
                if event_time // 1000 >= next_second:
                    top_bids, top_asks = order_book.get_top_levels()

                    # Capture the top 10 prices and volumes of bids and asks
                    bid_prices = [price for price, _ in top_bids] + [0] * (10 - len(top_bids))  # Fill with 0 if less than 10 levels
                    bid_volumes = [qty for _, qty in top_bids] + [0] * (10 - len(top_bids))

                    ask_prices = [price for price, _ in top_asks] + [0] * (10 - len(top_asks))
                    ask_volumes = [qty for _, qty in top_asks] + [0] * (10 - len(top_asks))

                    # Append to data list (timestamp + bid prices + bid volumes + ask prices + ask volumes)
                    data_list.append(
                        [current_second] + bid_prices + bid_volumes + ask_prices + ask_volumes
                    )

                    # Move to the next second
                    current_second = event_time // 1000
                    next_second = current_second + 1

    # Create a DataFrame
    columns = (
        ['timestamp'] +
        [f'bid_price_{i+1}' for i in range(10)] +
        [f'bid_volume_{i+1}' for i in range(10)] +
        [f'ask_price_{i+1}' for i in range(10)] +
        [f'ask_volume_{i+1}' for i in range(10)]
    )
    df = pd.DataFrame(data_list, columns=columns)
    
    return df


def compute_quantile_stripe_diff(log_prices, weights, alpha_start=0.0, alpha_end=1.0):
    # Compute the quantiles for the start and end
    quantile_start = np.quantile(log_prices, alpha_start)
    quantile_end = np.quantile(log_prices, alpha_end)

    # Mask values based on the quantile range
    mask = (log_prices >= quantile_start) & (log_prices <= quantile_end)
    
    # Take the mean of the values in this range, weighted by the given weights
    values_in_range = log_prices[mask]
    weights_in_range = weights[mask]
    
    if np.sum(weights_in_range) > 0:
        weighted_mean = np.average(values_in_range, weights=weights_in_range)
    else:
        weighted_mean = 0  # Handle case where no values fall in the range
    
    return weighted_mean

# Updated function to compute all indicators, including x_mean and x_median
def compute_indicators(df, weights_a=None, weights_b=None, alpha_start=0.0, alpha_end=1.0):
    if weights_a is None:
        weights_a = np.ones(10)
    if weights_b is None:
        weights_b = np.ones(10)
    
    indicators = []

    for _, row in df.iterrows():
        ask_prices = np.array([row[f'ask_price_{i+1}'] for i in range(10)])
        ask_volumes = np.array([row[f'ask_volume_{i+1}'] for i in range(10)])

        bid_prices = np.array([row[f'bid_price_{i+1}'] for i in range(10)])
        bid_volumes = np.array([row[f'bid_volume_{i+1}'] for i in range(10)])

        weighted_ask_volumes = weights_a * ask_volumes
        weighted_bid_volumes = weights_b * bid_volumes
        
        ask_L1_norm = np.sum(np.abs(weighted_ask_volumes))
        bid_L1_norm = np.sum(np.abs(weighted_bid_volumes))

        log_ask_prices = np.log(ask_prices + 1e-9)
        log_bid_prices = np.log(bid_prices + 1e-9)

        if (ask_L1_norm + bid_L1_norm) != 0:
            vol_diff = (ask_L1_norm - bid_L1_norm) / (ask_L1_norm + bid_L1_norm)
        else:
            vol_diff = 0

        avg_ask_log_price = np.mean(log_ask_prices)
        avg_bid_log_price = np.mean(log_bid_prices)
        avg_log_price_diff = avg_ask_log_price - avg_bid_log_price

        if (ask_L1_norm + bid_L1_norm) != 0:
            vol_weighted_log_price_diff = (ask_L1_norm * avg_ask_log_price - bid_L1_norm * avg_bid_log_price) / (ask_L1_norm + bid_L1_norm)
        else:
            vol_weighted_log_price_diff = 0

        stripe_diff_asks = compute_quantile_stripe_diff(log_ask_prices, weighted_ask_volumes, alpha_start, alpha_end)
        stripe_diff_bids = compute_quantile_stripe_diff(-log_bid_prices, weighted_bid_volumes, alpha_start, alpha_end)
        diff_stripes = stripe_diff_asks + stripe_diff_bids
        
        if (ask_L1_norm + bid_L1_norm) != 0:
            vol_weighted_diff_stripes = (ask_L1_norm * stripe_diff_asks - bid_L1_norm * stripe_diff_bids) / (ask_L1_norm + bid_L1_norm)
        else:
            vol_weighted_diff_stripes = 0

        # Estimating the market prices: Mean and Median of the combined orderbook
        combined_prices = np.concatenate([ask_prices, bid_prices])
        combined_volumes = np.concatenate([ask_volumes, bid_volumes])
        
        x_mean = np.average(combined_prices, weights=combined_volumes)  # Weighted mean
        x_median = np.median(combined_prices)  # Median
        
        ask_mean_diff = np.log(x_mean) - compute_quantile_stripe_diff(log_ask_prices, weighted_ask_volumes, 0, 0.01)
        ask_median_diff = np.log(x_median) - compute_quantile_stripe_diff(log_ask_prices, weighted_ask_volumes, 0, 0.01)

        bid_mean_diff = np.log(x_mean) - compute_quantile_stripe_diff(log_bid_prices, weighted_bid_volumes, 0, 0.01)
        bid_median_diff = np.log(x_median) - compute_quantile_stripe_diff(log_bid_prices, weighted_bid_volumes, 0, 0.01)

        indicators.append({
            'timestamp': row['timestamp'],
            'vol_diff': vol_diff,
            'avg_log_price_diff': avg_log_price_diff,
            'vol_weighted_log_price_diff': vol_weighted_log_price_diff,
            'diff_stripes': diff_stripes,
            'vol_weighted_diff_stripes': vol_weighted_diff_stripes,
            'ask_mean_diff': ask_mean_diff,
            'ask_median_diff': ask_median_diff,
            'bid_mean_diff': bid_mean_diff,
            'bid_median_diff': bid_median_diff
        })

    return pd.DataFrame(indicators).set_index('timestamp')
