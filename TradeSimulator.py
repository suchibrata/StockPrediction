import pandas as pd
from math import floor
N_STEPS=15
CAPITAL=100000

class TradeSimulator:
    def __init__(self,pred_filename,transaction_filename):
        self.pred_df=pd.read_csv(pred_filename)
        self.transaction_filename=transaction_filename
        self.next_index=0
        self.trans_df=pd.DataFrame()
        self.last_transaction=None
        self.stock_holding=0
        self.capital=CAPITAL

    def get_next_data(self):
        if self.pred_df.shape[0] > self.next_index:
            row = self.pred_df.iloc[self.next_index]
            row = row.to_dict()
            #print(row)
            # row=row[0]
            self.next_index += 1
        else:
            row = None
        return row

    def get_trade_decision(self,row):
        pred_prices=[]
        for idx in range(N_STEPS):
            pred_prices.append(row['T{}'.format(idx+1)])

        min_p=min(pred_prices)
        max_p=max(pred_prices)

        curr_p=row['Close']
        dec='HOLD'
        if self.stock_holding==0:
            if curr_p > max_p:
                dec='SELL'
            elif curr_p < min_p:
                dec='BUY'
        elif self.stock_holding >0:
            hold_p=self.last_transaction[0]
            if (1-hold_p/curr_p) >=0.01:
                dec='STOPLOSS-SELL'
            elif curr_p > max_p:
                dec='HOLD'
            elif curr_p < min_p:
                dec='SELL'
        elif self.stock_holding <0:
            hold_p = self.last_transaction[0]
            if (1 - curr_p / hold_p)>=0.01:
                dec = 'STOPLOSS-BUY'
            elif curr_p > max_p:
                dec = 'BUY'
            elif curr_p < min_p:
                dec = 'HOLD'

        return dec

    def trade(self):

        while True:
            row=self.get_next_data()
            if not row:
                break
            dec=self.get_trade_decision(row)
            curr_p=row['Close']

            if self.stock_holding==0:
                if dec=='BUY':
                    units=floor(self.capital/curr_p)
                    money_flow=-units*curr_p
                    self.trans_df=self.trans_df.append({'price':curr_p,'units':units,'decision':dec,'money_flow':money_flow},
                                                       ignore_index=True)
                    self.capital+=money_flow
                    self.last_transaction=(curr_p,units)
                    self.stock_holding+=units
                    print("{} - {} - {} - {}".format(dec,units,curr_p,self.capital))
                elif dec=='SELL':
                    units = floor(self.capital / curr_p)
                    money_flow = units * curr_p
                    self.trans_df = self.trans_df.append({'price': curr_p, 'units': units, 'decision': dec, 'money_flow': money_flow},
                        ignore_index=True)
                    self.capital += money_flow
                    self.last_transaction = (curr_p, -units)
                    self.stock_holding -= units
                    print("{} - {} - {} - {}".format(dec, units, curr_p, self.capital))

            elif self.stock_holding>0:
                if dec in ('STOPLOSS-SELL','SELL'):
                    units = self.last_transaction[1]
                    money_flow = units * curr_p
                    self.trans_df = self.trans_df.append({'price': curr_p, 'units': units, 'decision': dec, 'money_flow': money_flow},
                        ignore_index=True)
                    self.capital += money_flow
                    self.last_transaction = None
                    self.stock_holding-= units
                    print("{} - {} - {} - {}".format(dec, units, curr_p, self.capital))
            elif self.stock_holding<0:
                if dec in ('STOPLOSS-BUY', 'BUY'):
                    units = self.last_transaction[1]
                    money_flow = units * curr_p
                    self.trans_df = self.trans_df.append({'price': curr_p, 'units': abs(units), 'decision': dec, 'money_flow': money_flow},
                        ignore_index=True)
                    self.capital += money_flow
                    self.last_transaction = None
                    self.stock_holding -= units
                    print("{} - {} - {} - {}".format(dec, units, curr_p, self.capital))

        if self.stock_holding !=0:
            dec = 'BUY' if self.stock_holding <0 else 'SELL'
            units=self.stock_holding
            money_flow = units * curr_p
            self.trans_df = self.trans_df.append(
                {'price': curr_p, 'units': abs(units), 'decision': dec, 'money_flow': money_flow},
                ignore_index=True)
            self.capital += money_flow
            self.last_transaction = None
            self.stock_holding -= units
            print("{} - {} - {} - {}".format(dec, units, curr_p, self.capital))

        self.trans_df.to_csv(self.transaction_filename)


    def calculate_pl(self):
        profit=(self.capital/CAPITAL-1)*100
        print("profit is {} %".format(profit))


if __name__=='__main__':
    trd=TradeSimulator(r'tata_motors_prediction_ns.csv',r'tata_motors_transactions.csv')
    trd.trade()
    trd.calculate_pl()















