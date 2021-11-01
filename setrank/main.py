import train
import predata
import argparse

def parser_init():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('address', type=str, help='input dataset address')
    parser.add_argument('lr', type=float, help='input lr')
    parser.add_argument('epochs', type=int, help='input epochs')
    parser.add_argument('device', type=str, help='input device')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_init()
    datas = predata.getdata(args.address)
    train.train(datas)
