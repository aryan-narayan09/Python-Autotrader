import mode1
import websocket, json
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException



#Websocket stream
SOCKET = "wss://fstream.binance.com/ws/adausdt_perpetual@continuousKline_1m"


#order related variables
symbol = "ADAUSDT"
balance = 0
contract_size = 0
lvg = 0
signal = -1
ppc = 0
order_placed = False



#configuring binance client
client = Client("API_PRIVATE", "API_PUBLIC")


#function to place buy order
def buy_order():
    global prev_close, contract_size, lvg, ppc, order_placed, high

    #getting portfolio value
    acc_balances = client.futures_account_balance()
    usdt_balance = next(item for item in acc_balances if item["asset"] == "USDT")
    balance = json.loads(usdt_balance['balance'])
    balance = int(balance)

    #calculating quantity to buy for risk management
    contract_size = round((balance*0.02)/(0.02*0.02))
    
    #calculating leverage
    lvg = (contract_size*close)/balance

    close = json.loads(prev_close)     
    buy_price = high
    print(buy_price)

    
    try:
        print("Sending buy order")
        client.futures_create_order(   
        orderId = 1,
        symbol = symbol,
        leverage = lvg,
        side = Client.SIDE_BUY,
        type = Client.FUTURE_ORDER_TYPE_STOP_MARKET,   
        stopPrice = buy_price,
        quantity = contract_size)
        print("Buy order placed")


        print("Sending Take Profit order")
        client.futures_create_order(  
        orderId = 2,
        symbol = symbol,
        leverage = lvg,
        side = Client.SIDE_SELL,
        type = Client.FUTURE_ORDER_TYPE_TAKE_PROFIT,                     
        stopPrice = buy_price + (buy_price*(ppc/100)),
        price = buy_price + (buy_price*(ppc/100)),
        quantity = contract_size,
        reduceOnly = 'true')
        print("Take Profit order placed")  

        order_placed = True 

        print("Sending SL order")
        client.futures_create_order(   
        orderId = 1,
        symbol = symbol,
        leverage = lvg,
        side = Client.SIDE_SELL,
        type = Client.FUTURE_ORDER_TYPE_STOP_MARKET, 
        stopPrice = buy_price-(contract_size*close)*0.02,
        reduceOnly = 'true',
        quantity = contract_size)
        print("Stop Loss order placed")
        print("\n ------------------------------- \n")
        exit()

    except BinanceAPIException as e:
        print(e)
  
    except BinanceOrderException as e:
        print(e)




#opening connection to websocket
def on_open(ws):
    global balance
    print("\n ---------------------------------------------- \n")
    print("Opened Connection")

    


#closing connection to websocket
def on_close(ws):
    print("Closed Connection")


#Executes everytime message received from websocket
def on_message(ws, message):

    global contract_size, order_placed, signal, ppc, high

    json_message = json.loads(message)
  
    #defining each candle and their highs and lows
    candle = json_message['k']
    is_candle_closed = candle['x']
    low = candle['l']
    high = candle['h']
    close = candle['c']
    open = candle['o']
    volume = candle['v']


    #checking logic after each candle close
    if is_candle_closed:
        
        global prev_low, prev_close, contract_size, lvg
        prev_low = low
        prev_close = close

        #extracting parameters from kline data
        candle = json_message['k']
        is_candle_closed = candle['x']
        open_time = candle['t']
        open = candle['o']
        high = candle['h']
        low = candle['l']
        close = candle['c']
        volume = candle['v']
        close_time = candle['T']
        quote_volume = candle['q']
        count = candle['n']
        taker_buy_volume = candle['V']
        taker_buy_quote_volume = candle['Q']
        ignore = candle['B']
        


        #printing candle details
        print("Open Time: {}".format(open_time))
        print("Open: {}".format(open))
        print("High:{}".format(high))
        print("Low: {}".format(low))
        print("Close:{}".format(close))
        print("Volume:{}".format(volume))
        print("Close Time:{}".format(close_time))
        print("Quote Volume:{}".format(quote_volume))
        print("Count:{}".format(count))
        print("Taker Buy Volume:{}".format(taker_buy_volume))
        print("Taker Buy Quote Volume:{}".format(taker_buy_quote_volume))
        print("Ignore:{}".format(ignore))
        print("\n---------------------------------------------\n")

        mode1.feedData()
        
        if mode1.signal == 1:
            buy_order()




#configuring websocket connection and running
ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()