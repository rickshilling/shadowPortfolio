'''
this program automates the order entry process for AAII shadow protfolio.

It does so by reading in the order entry CSV file with the ticker, command, and share size. It
will then get today's best quote for each ticker, print that on the terminal, calculate the 
approximate cost of the BUY LMT orders read from the CSV file, and also print that for user confirmation.
If user confirms, the orders will be sent to the exchange. 

touch
'''
import configparser, os, json
import argparse
import pandas as pd
import pyetrade
from datetime import datetime
from pathlib import Path

class BatchOrder():
    def __init__(self, config, orders, sandbox=True) -> None:
        
        consumer_key    = config["DEFAULT"]["CONSUMER_KEY"]
        consumer_secret = config["DEFAULT"]["CONSUMER_SECRET"]
        oauth = pyetrade.ETradeOAuth(consumer_key, consumer_secret)
        
        print('Checking Tokens')
        tokens = self.getCachedTokens()
        if None == tokens:
            # couldn't find good tokens so go get new ones
            print("Go to the following address, log in to your eTrade account and copy the Code it presents you.")
            print(oauth.get_request_token())  # Use the printed URL

            verifier_code = input("Enter verification code: ")
            tokens = oauth.get_access_token(verifier_code)
            
            # save these tokens so not to bug the user so many times. Get new ones every hour. Could try longer.
            if None == tokens:
                print(f'Error getting tokens')
                return
            else:
                self.saveTokens(tokens)
        
        self.accounts = pyetrade.ETradeAccounts(consumer_key,
                                                consumer_secret,
                                                tokens['oauth_token'],
                                                tokens['oauth_token_secret'])
        
        # get accounts for this user
        accts = self.accounts.list_accounts()['AccountListResponse']['Accounts']['Account']
        print(f'Num total Account: {len(accts)}')
            
        act_accts = [x for x in accts if x['accountStatus'] == 'ACTIVE']
        print(f'Num active accounts: {len(act_accts)}')

        self.printAccounts(act_accts) # print summary

        # FIX
        # self.getAccountBal(act_accts)

    def saveTokens(self, tokens):
        dir = Path('.cache')
        dir.mkdir(exist_ok=True)
        ts = datetime.today().strftime('%Y-%m-%d-%H')
        f = open(os.path.join(dir, ts + ".json"), 'w')
        json.dump(tokens, f)
    
    def getCachedTokens(self):
        ts = datetime.today().strftime('%Y-%m-%d-%H')
        dir = Path('.cache')
        if dir.exists():
            fname = Path(os.path.join(dir, ts + ".json"))
            if fname.exists():
                f = open(fname, 'r')
                js = json.load(f)
                print(f'Loaded keys={js}')
                return js
        else: return None

    def printAccounts(self, accts):
        if self.accounts:
            print(f'---- ACCOUNT SUMMARY ----')
            print('Name         Desc             Type         Mode   ID')
            print('------------------------------------------------------------')
            for x in accts:
                name = x['accountName']
                desc = x['accountDesc']
                _type = x['accountType']
                mode = x['accountMode']
                id = x['accountId']
                print(f'{name:12s} {desc:16s} {_type:12s} {mode:6s} {id:12s}')
                
    def getAccountBal(self, accts):
        for x in accts:
            id = x['accountId']
            ret = self.accounts.get_account_balance(id)
            print(f'ret={ret}\n')

def main():
    # Parse args
    parser = argparse.ArgumentParser(prog="batch order")
    parser.add_argument('--config', default='config.ini')
    parser.add_argument('--csv_orders', type=str)
    args = parser.parse_args()
    print(args)

    # load configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Load order from CSV
    df = pd.read_csv(args.csv_orders)
    print(df)

    # Init batch order class and print account balance
    batchOrder = BatchOrder(config, df)

if __name__ == '__main__':
    main()
