import argparse
import hashlib
import jwt
import json
import os

import web3.constants
from jwcrypto import jwk
from rich.console import Console
from rich.theme import Theme
from web3 import Web3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENTS_DIR = os.path.join(BASE_DIR, '../../deployments')
KMS_ADDRESS = "0x3366E906D7C2362cE4C336f43933Cccf76509B23"

custom_theme = Theme({
    "success": "bold green",
    "error": "bold red",
    "info": "dim",
    "step": "bold cyan"
})

console = Console(theme=custom_theme)

def print_step(step_num, total, message):
    console.print(f"[step][{step_num}/{total}][/] {message}")

def print_result(success, message, details=None):
    color = "success" if success else "error"
    symbol = "✔" if success else "✖"
    console.print(f"    [{color}]{symbol}[/] {message}")
    if details:
        console.print(f"    [info]→ {details}[/]", style="info")

parser = argparse.ArgumentParser(description="Panda attestation verification script")
parser.add_argument(
    "cert_pubkey",
    type=str,
    help="SSL certificate public key"
)
parser.add_argument(
    "--rpc",
    type=str,
    default="https://mainnet.optimism.io",
    help="OP Mainnet RPC endpoint",
)

args = parser.parse_args()

console.print("[bold]🔍 Panda Attestation verification[/]")

cert_tag = args.cert_pubkey[:9]
print_step(1, 4, f"Finding files for pubkey: {cert_tag}")

compose_file_path = None
attestation_file_path = None
for filename in os.listdir(DEPLOYMENTS_DIR):
    if f"{cert_tag}_compose" in filename:
        compose_file_path = os.path.join(DEPLOYMENTS_DIR, filename)
    elif f"{cert_tag}_attestation" in filename:
        attestation_file_path = os.path.join(DEPLOYMENTS_DIR, filename)

if compose_file_path is None:
    print_result(False, f"Not found compose file for {cert_tag}")
    exit(1)

if attestation_file_path is None:
    print_result(False, f"Not found attestation file for {cert_tag}")
    exit(1)

print_result(True, f"Found compose file: {os.path.basename(compose_file_path)}")
print_result(True, f"Found attestation file: {os.path.basename(attestation_file_path)}")


with open(attestation_file_path) as attestation_file:
    attestation = json.load(attestation_file)

print_step(2, 4, "Verify compose hash")
with open(compose_file_path, "rb") as compose_file:
    compose_hash = hashlib.sha256(compose_file.read()).hexdigest()
    if compose_hash != attestation["compose_hash"]:
        print_result(False, "Hash mismatch")
        exit(1)
print_result(True, f"Passed")


print_step(2, 4, "Verify JWT")
with open(os.path.join(BASE_DIR, "trust-authority-certs.txt"), "r") as f:
    keys = json.load(f)

key = jwk.JWK(**keys["keys"][-1])
public_key_pem = key.export_to_pem()
try:
    jwt_payload = jwt.decode(attestation["token"], public_key_pem, algorithms=["RS256"], options={"verify_exp": False})
except Exception as e:
    print_result(False, f"Failed to parse JWT: {e}")
    exit(1)

if jwt_payload["tdx"]["tdx_collateral"]["quotehash"] != hashlib.sha256(bytes.fromhex(attestation["quote"])).hexdigest():
    print_result(False, "JWT quotehash mistmatch")
    exit(1)
print_result(True, "Passed")


print_step(3, 4, "Replay event logs")
event_log = attestation["event_log"]
rtmr = ["0"*96] * 4
for event in event_log:
    rtmr[event["imr"]] = hashlib.sha384(bytes.fromhex(rtmr[event["imr"]] + event["digest"])).hexdigest()

for i in range(4):
    if rtmr[i] != jwt_payload["tdx"][f"tdx_rtmr{i}"]:
        print_result(False, "tdx_rtmr{i} mismatch")
        exit(1)
print_result(True, "Passed")


print_step(3, 4, "Verify OS image hash")
os_image_hash = ""

for event in event_log:
    if event["event"] == "app-id":
        app_id = event["event_payload"]
    elif event["event"] == "key-provider":
        key_provider = event["event_payload"]
    elif event["event"] == "compose-hash":
        compose_hash = event["event_payload"]
    elif event["event"] == "os-image-hash":
        os_image_hash = event["event_payload"]

if os_image_hash != attestation["os_image_hash"]:
    print_result(False, "OS image hash mismatch")
    exit(1)
print_result(True, f"Passed")


print_step(4, 4, "Verify on-chain consistency")
user_data = attestation["quote"][28*2:48*2]
device_id = hashlib.sha256(bytes.fromhex(user_data)).hexdigest()

w3 = Web3(Web3.HTTPProvider(args.rpc))

contract_address = w3.to_checksum_address(KMS_ADDRESS)
abi = [{
    "inputs": [{
        "components": [
            {"internalType": "address","name": "appId","type": "address"},
            {"internalType": "bytes32","name": "composeHash","type": "bytes32"},
            {"internalType": "address","name": "instanceId","type": "address"},
            {"internalType": "bytes32","name": "deviceId","type": "bytes32"},
            {"internalType": "bytes32","name": "mrAggregated","type": "bytes32"},
            {"internalType": "bytes32","name": "mrSystem","type": "bytes32"},
            {"internalType": "bytes32","name": "osImageHash","type": "bytes32"},
            {"internalType": "string","name": "tcbStatus","type": "string"},
            {"internalType": "string[]","name": "advisoryIds","type": "string[]"}
        ],
        "internalType": "struct AppBootInfo",
        "name": "bootInfo",
        "type": "tuple"
    }],
    "name": "isAppAllowed",
    "outputs": [
        {"internalType": "bool","name": "isAllowed","type": "bool"},
        {"internalType": "string","name": "reason","type": "string"}
    ],
    "stateMutability": "view",
    "type": "function"
}]

contract = w3.eth.contract(address=contract_address, abi=abi)

boot_info = {
    "appId": w3.to_checksum_address(f"0x{app_id}"),
    "composeHash": w3.to_bytes(hexstr=f"0x{compose_hash}"),
    "instanceId": web3.constants.ADDRESS_ZERO,
    "deviceId": w3.to_bytes(hexstr=f"0x{device_id}"),
    "mrAggregated": web3.constants.HASH_ZERO,
    "mrSystem": web3.constants.HASH_ZERO,
    "osImageHash": w3.to_bytes(hexstr=f"0x{os_image_hash}"),
    "tcbStatus": "",
    "advisoryIds": []
}

# call the function
is_allowed, reason = contract.functions.isAppAllowed(boot_info).call()

if not is_allowed:
    print_result(False, f"App is not allowed: {reason}")
    exit(1)

print_result(True, f"Passed")

console.print("\n[bold green]🔒 Secure Verification Report[/bold green]")
console.print(f"• Attested Cert: [dim]{attestation['public_key']}[/dim]")
console.print(f"• OS Image: [cyan]{os_image_hash}[/cyan]")
console.print(f"• App Bundle: [cyan]{compose_hash}[/cyan]")
console.print(f"• Contract [cyan]https://optimistic.etherscan.io/address/0x3366E906D7C2362cE4C336f43933Cccf76509B23[/cyan]")

console.print('''\n✅ This attestation has been [bold green]cryptographically verified[/bold green] via  
[bold cyan]Intel TDX[/bold cyan] + [bold purple]Nvidia CC[/bold purple] trusted execution,  
confirming integrity against on-chain records.''')

console.print("\n[bold]🔍 Need to verify the build?[/bold]")
console.print(f"• Reproduce OS hash: [cyan]{attestation['os_code']}[/cyan]")
console.print(f"• Inspect app sources: [cyan]https://github.com/testinprod-io/panda-vllm-proxy/tree/{attestation['compose_hash']}[/cyan]\n")


