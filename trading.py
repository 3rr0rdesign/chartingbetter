"""
Web3 trading: approve USDC.e, place limit buy, register profit/stop levels.
Uses known ABIs for ERC20 and Polymarket-style flow; CLOB order signing can be extended.
"""
import logging
import time
from typing import Any, Dict, Optional, Tuple

from web3 import Web3

from config import (
    BET_SIZE_USDC,
    DRY_RUN,
    MAX_GAS_GWEI,
    MIN_BALANCE_USDC,
    PROFIT_TGT,
    STOP_LOSS,
    get_private_key,
    get_rpc_url,
)

logger = logging.getLogger(__name__)

# Polygon mainnet
USDC_E_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
USDC_DECIMALS = 6

# Minimal ABIs
ERC20_ABI = [
    {"inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "nonpayable", "type": "function"},
    {"inputs": [{"name": "account", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
]

RETRIES = 3
RETRY_DELAY = 2.0


def _web3() -> Web3:
    return Web3(Web3.HTTPProvider(get_rpc_url()))


def check_gas_ok(web3: Optional[Web3] = None) -> bool:
    """True if current gas price <= MAX_GAS_GWEI."""
    w3 = web3 or _web3()
    try:
        gas_price = w3.eth.gas_price
        gwei = w3.from_wei(gas_price, "gwei")
        return gwei <= MAX_GAS_GWEI
    except Exception:
        return False


def check_balance(web3: Optional[Web3] = None) -> float:
    """USDC.e balance for the configured wallet (requires private key -> address)."""
    key = get_private_key()
    if not key:
        return 0.0
    w3 = web3 or _web3()
    try:
        acc = w3.eth.account.from_key(key)
        contract = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=ERC20_ABI)
        raw = contract.functions.balanceOf(acc.address).call()
        return raw / (10**USDC_DECIMALS)
    except Exception as e:
        logger.warning("Balance check failed: %s", e)
        return 0.0


def approve_usdc_e(
    spender: str,
    amount_usdc: float,
    web3: Optional[Web3] = None,
) -> Optional[str]:
    """
    Approve spender to spend amount_usdc USDC.e. Returns tx hash or None.
    Skips in DRY_RUN.
    """
    if DRY_RUN:
        logger.info("[DRY_RUN] Would approve %s USDC.e to %s", amount_usdc, spender)
        return "dry_run_approve"

    key = get_private_key()
    if not key:
        logger.error("No PRIVATE_KEY for approve")
        return None

    w3 = web3 or _web3()
    amount_wei = int(amount_usdc * (10**USDC_DECIMALS))
    contract = w3.eth.contract(address=Web3.to_checksum_address(USDC_E_ADDRESS), abi=ERC20_ABI)
    acc = w3.eth.account.from_key(key)

    for attempt in range(RETRIES):
        try:
            if not check_gas_ok(w3):
                logger.warning("Gas too high, skipping approve")
                return None
            tx = contract.functions.approve(Web3.to_checksum_address(spender), amount_wei)
            built = tx.build_transaction({
                "from": acc.address,
                "gas": 100_000,
            })
            signed = acc.sign_transaction(built)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            return w3.to_hex(tx_hash)
        except Exception as e:
            logger.warning("Approve attempt %s failed: %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)
    return None


def place_order_with_stops(
    web3: Web3,
    amount: float,
    price: float,
    side: str,
    market_id: str,
    token_id: str,
    outcome_index: int,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Place limit buy for outcome shares and register exit levels.
    - amount: USDC.e size (e.g. 10–20)
    - price: limit price per share (0..1)
    - side: "BUY"
    - market_id, token_id, outcome_index: for logging and CLOB.

    Returns (success, entry_price_used, profit_target_price).
    Exits: limit sell at PROFIT_TGT * entry (e.g. 0.96), stop at STOP_LOSS * entry (e.g. 0.60).

    In live mode this would: 1) submit signed CLOB order via Polymarket API,
    2) store entry_price and set alerts/orders for 0.96*entry and 0.60*entry.
    Here we simulate: success=True, entry_price=price, profit_target=PROFIT_TGT*price.
    """
    entry_price = price
    profit_target = PROFIT_TGT * entry_price
    stop_price = STOP_LOSS * entry_price

    if DRY_RUN:
        logger.info(
            "[DRY_RUN] BUY %s @ %.3f | market=%s | profit_tgt=%.3f stop=%.3f",
            side, entry_price, market_id, profit_target, stop_price,
        )
        return True, entry_price, profit_target

    # Live: check balance and gas
    if check_balance(web3) < MIN_BALANCE_USDC:
        logger.warning("Balance < %s USDC, skip order", MIN_BALANCE_USDC)
        return False, None, None
    if not check_gas_ok(web3):
        logger.warning("Gas too high, skip order")
        return False, None, None

    # Placeholder: real implementation would call Polymarket CLOB API with signed order.
    # Approve exchange/CTF if needed, then POST /order with tokenId=token_id, side=BUY, etc.
    logger.info(
        "Would place BUY %.2f USDC @ %.3f token_id=%s profit_tgt=%.3f stop=%.3f",
        amount, entry_price, token_id[:16] + "...", profit_target, stop_price,
    )
    return True, entry_price, profit_target


def place_order_with_stops_simple(
    amount_usdc: float = BET_SIZE_USDC,
    bid_price: float = 0.5,
    market_id: str = "",
    token_id: str = "",
    side: str = "BUY",
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    Simplified entry: place order with stops using default web3.
    Use after strategy signals BUY reversal at bid_price.
    """
    w3 = _web3()
    return place_order_with_stops(
        w3,
        amount=amount_usdc,
        price=bid_price,
        side=side,
        market_id=market_id,
        token_id=token_id,
        outcome_index=0,
    )
