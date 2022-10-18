from http import client
from itertools import count
from venv import create
import hvac 
import random 

def makeConn():
    hvc_client = client = hvac.Client(url='http://127.0.0.1:8200', token='hvs.mGMlKHHLScVv39ejnLAJOpvz' ) 
    return hvc_client 

def storeSecret( client,  secr1 , cnt  ):
    secret_path     = 'SECRET_PATH_' + str( cnt  )
    create_response = client.secrets.kv.v2.create_or_update_secret(path=secret_path, secret=dict(password =  secr1 ) )
    #print( type( create_response ) )
    #print( dir( create_response)  )

def retrieveSecret(client_, cnt_): 
    secret_path        = 'SECRET_PATH_' + str( cnt_  )
    read_response      = client_.secrets.kv.read_secret_version(path=secret_path) 
    secret_from_vault  = read_response['data']['data']['password']
    print('The secret we have obtained:')
    print(secret_from_vault)

if __name__ == '__main__':
    clientObj    =  makeConn()
    secretlist = [ 'root_user', 'test_password', 'ghp_ahAyHoRwoQ', 'MTIzANO=' , 't5f28U'] 
    
    for i, secret in enumerate(secretlist):
        print('The secret we want to store:', secret)
        print('='*50)
        storeSecret( clientObj,   secret, i )
        print('='*50)
        retrieveSecret( clientObj,  i )
        print('='*50)