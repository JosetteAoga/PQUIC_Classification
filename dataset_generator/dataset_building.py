import subprocess
import shlex
import sys
import time
import random
from shutil import copyfile, rmtree
import csv
from scapy.sendrecv import AsyncSniffer
from scapy.utils import RawPcapReader, wrpcap, rdpcap
from scapy.packet import Packet
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP
import numpy as np
import os
import argparse


outdoc =  "path" # location where the files created and to be used are
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="plugin name") # name in lowercase eg: fec
parser.add_argument("-x", help="recursivity deep", type=int)
parser.add_argument("-o", help="out file name without extension")
args = parser.parse_args()

out_name = outdoc + (args.o + ".csv" if args.o is not None else "data_flow.csv")
iteration = args.x if args.x is not None else 1

classe = 0
plugin = args.p
if plugin == "fec":
    classe = 0
elif plugin == "monitoring":
    classe = 1
elif plugin == "multipath":
    classe = 2
else:
    classe = 3


def client_pquicdemo(filename, host, g):
    return "picoquicdemo -l " + filename + " -4 -G " + g + host + " 4443"

def generate_dump_command_string(server_ip):
    return "tshark -i wlo1 -f 'udp and host {} and port 4443' -w net.pcap".format(server_ip)

def generate_tstat_command_string():
    return "tstat -s trace1 -N net.conf -s " + outdoc + " net.pcap"

def dataset_generator(server_ip, file, variance_out, variance_in, variance, logs, output):
    to_return = []
    print("chem", file)
    with open(file, "r") as capture:
        print("dataset generator")
        for i, line in enumerate(capture):
            if i != 0:
                a = line.strip("\n").split(" ")
                if server_ip in a[0]:
                    a = a[9:17+1] + a[:8+1] + a[18:]  # re-order features when client and server are inverted
                print(a + [classe])
                print(a[4], a[5], a[13], a[14])
                try:
                    average_client = round(int(a[4]) / int(a[5]), 4)
                    average_server = round(int(a[13]) / int(a[14]), 4)
                except:
                    continue
                to_return.append(a + [average_client, average_server, variance_out, variance_in, variance, logs, output, classe])
               
    return to_return

with open(out_name, "w") as outfile:
    host = "server_hostname" # put here the hostname of your server
    server_ip = "0.0.0.0" 
    filename = "file.txt" # file to save the log
    
    writer = csv.writer(outfile)
    # features of the dataset
    writer.writerow(['c_ip', 'c_port', 'c_first_abs', 'c_durat', 'c_bytes_all', 'c_pkts_all', 'c_isint',
                     'c_iscrypto', 'c_type', 's_ip', 's_port', 's_first_abs', 's_durat', 's_bytes_all',
                     's_pkts_all', 's_isint', 's_iscrypto', 's_type', 'fqdn', 'quic_SNI', 'quic_UA', 'quic_CHLO',
                   'quic_REJ', 'average_c_bytes', 'average_s_bytes', 'variance_out', 'variance_in', 'variance', 'logs', 'console_out', 'classe'])
    outfile.flush()
    
    random.seed(1)
    for n, g in enumerate(random.sample(range(10000, 100000), 2)): # Take 10 different g in the range

        for m in range(0, iteration):

            print("launch packets capturing")
            # The following instruction, launch tshark command
            dump_proc = subprocess.Popen(shlex.split(generate_dump_command_string(server_ip)), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
            
            # launch the picoquicdemo command from the client side
            print("launch picoquicdemo command  from client")
            client_proc = subprocess.Popen(shlex.split(client_pquicdemo(filename, host, str(g))), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
            out, err = client_proc.communicate() # Save the output and the error raise by the command
            output = err + ' ' + out
            
            with open(filename, "r") as f:
                logs = f.read().rstrip()
        
            client_proc.wait()
            print("pquic finished")
        
            dump_proc.terminate()
            print("packets capturing stopped\n")
            
            print("get packets dumped from pcap file produce by tshark command")
            pkts = rdpcap('net.pcap')
            print("done\n")

            print("Start computing variance")
            # read the pcap to get the length of each packet
            pkts_length_s = []
            pkts_length_c = []
            for i, pkt in enumerate(pkts):
                if pkt[IP].dst == server_ip:
                    length = pkt[IP].len
                    pkts_length_c.append(length)
                else:
                    length = pkt[IP].len
                    pkts_length_s.append(length)
            # compute the variance
            try:
                variance_c = np.var(np.array(pkts_length_c))
                variance_s = np.var(np.array(pkts_length_s))
                variance = np.var(np.array(pkts_length_c + pkts_length_s))
            except:
                variance_c = 0
                variance_s = 0
                variance = 0
            print("variance computed\n")

            # launch tstat by passing the pcap file as input
            # extract features with tstat and add the variance
            # save the features
            print("launch tsat on the pcap file")
            tstat_proc = subprocess.Popen(shlex.split(generate_tstat_command_string()), shell=False)
            tstat_proc.wait()
            print("tstat finished")

            # Take the features created by Tstat from UDP log files
            file_out = os.listdir(outdoc)
            directory = [f for f in file_out if "out" in f][0]
            print("directory created by tstat", directory)
            log_udp = outdoc + "{}/log_udp_complete".format(directory)
            print("full path of the directory", log_udp, "\n")
            
            
            print("read tstat file to extract features")
            for line in dataset_generator(server_ip, log_udp, variance_c, variance_s, variance, logs, output):
                writer.writerow(line)
                outfile.flush()
            print("features added to csv\n")
            
            print("iteration: G:", n+1, "it:", m+1, "==>", (n*iteration+m)+1)
            print("DONE!!!\n\n\n")
        
            timestr = time.strftime("%Y%m%d_%H%M%S")
            subprocess.check_call(["rm", "-r", outdoc + directory])
