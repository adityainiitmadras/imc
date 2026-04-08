"""
Wall-mid-centric trader for Prosperity 4 Tutorial Round.

Key insights applied from top Prosperity 2/3 teams:
- Frankfurt Hedgehogs (2nd, Prosperity 3): Kelp traded identically to Resin
  using "wall mid" (L2 market-maker mid) as fair value.
- Linear Utility (2nd, Prosperity 2): Position clearing via 0-EV trades to
  free up capacity for more profitable market-making.
- Alpha Animals (9th, Prosperity 3): Identified large-volume market maker
  quotes as the reliable fair value signal; small L1 orders are noise.

Data analysis confirms for Prosperity 4:
- EMERALDS: L1=9992/10008, L2=9990/10010. Wall mid always 10000. Spread=16.
- TOMATOES: L2 has 15-25 lot walls with spread=16. L1 has 5-10 lot noise
  traders sitting 1-2 ticks inside. L1 is ALWAYS strictly inside L2.
  Wall mid drifts but is locally stationary (autocorr=-0.17 vs L1 mid=-0.40).
  Narrow-spread L1 signals are adverse (accuracy 11%), NOT informed.
"""

import json
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple


EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"
LIMIT = 80


# ═══════════════════════════════════════════════════════════════
#  Helper: extract wall (L2) bid/ask from order book
# ═══════════════════════════════════════════════════════════════

def get_wall_level(
    book_levels: list,  # sorted [(price, vol), ...] — bids desc, asks asc
    side: str,          # "bid" or "ask"
) -> Optional[Tuple[int, int]]:
    """
    Return the 'wall' price level — the level with the largest volume.
    In Prosperity 4 tutorial data this is consistently L2 for TOMATOES
    and L1/L2 for EMERALDS.
    """
    if not book_levels:
        return None
    best = book_levels[0]
    for px, vol in book_levels:
        if abs(vol) > abs(best[1]):
            best = (px, vol)
    return (best[0], abs(best[1]))


class Trader:

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        if EMERALDS in state.order_depths:
            result[EMERALDS] = self.trade_emeralds(state)

        if TOMATOES in state.order_depths:
            result[TOMATOES] = self.trade_tomatoes(state)

        return result, conversions, ""

    # ══════════════════════════════════════════════════════════
    #  EMERALDS — Fixed fair=10000, take+clear+make
    # ══════════════════════════════════════════════════════════

    def trade_emeralds(self, state: TradingState) -> List[Order]:
        product = EMERALDS
        orders: List[Order] = []
        od = state.order_depths[product]
        position = state.position.get(product, 0)

        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])
        asks = sorted(od.sell_orders.items(), key=lambda x: x[0])

        if not bids or not asks:
            return orders

        fair = 10_000  # Always exactly 10000 — confirmed from data

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        wall_mid = (best_bid + best_ask) / 2.0

        # ── PASS 1: TAKING ─────────────────────────────────────
        # Buy anything offered below fair, sell anything bid above fair.
        # Also do 0-EV clearing: buy AT fair when short, sell AT fair when long.
        # (Linear Utility insight: frees position capacity for more MM trades)

        cur_pos = position

        for sp, sv in asks:
            sv = abs(sv)
            buy_room = LIMIT - cur_pos
            if buy_room <= 0:
                break
            if sp < fair:
                # Positive edge — take it all
                vol = min(sv, buy_room)
                orders.append(Order(product, sp, vol))
                cur_pos += vol
            elif sp == fair and cur_pos < 0:
                # 0-EV clearing: buy at fair to reduce short position
                vol = min(sv, abs(cur_pos), buy_room)
                if vol > 0:
                    orders.append(Order(product, sp, vol))
                    cur_pos += vol

        for bp, bv in bids:
            bv = abs(bv)
            sell_room = LIMIT + cur_pos
            if sell_room <= 0:
                break
            if bp > fair:
                # Positive edge — take it all
                vol = min(bv, sell_room)
                orders.append(Order(product, bp, -vol))
                cur_pos -= vol
            elif bp == fair and cur_pos > 0:
                # 0-EV clearing: sell at fair to reduce long position
                vol = min(bv, cur_pos, sell_room)
                if vol > 0:
                    orders.append(Order(product, bp, -vol))
                    cur_pos -= vol

        # ── PASS 2: MAKING ─────────────────────────────────────
        # Penny-jump the BBO (overbid L1 bid by 1, undercut L1 ask by 1)
        # but never cross fair.

        bid_price = best_bid + 1
        ask_price = best_ask - 1

        # Try to overbid the biggest visible bid if it's small enough
        for bp, bv in bids:
            overbid = bp + 1
            if abs(bv) > 1 and overbid < fair:
                bid_price = max(bid_price, overbid)
                break
            elif bp < fair:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in asks:
            undercut = sp - 1
            if abs(sv) > 1 and undercut > fair:
                ask_price = min(ask_price, undercut)
                break
            elif sp > fair:
                ask_price = min(ask_price, sp)
                break

        # Never cross fair
        bid_price = min(bid_price, fair - 1)
        ask_price = max(ask_price, fair + 1)

        remaining_buy = LIMIT - cur_pos
        remaining_sell = LIMIT + cur_pos

        if remaining_buy > 0:
            orders.append(Order(product, bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, ask_price, -remaining_sell))

        return orders

    # ══════════════════════════════════════════════════════════
    #  TOMATOES — Wall-mid market making (mirrors EMERALDS logic)
    #
    #  Key insight: TOMATOES has a two-layer book structure.
    #  L2 = big market maker (15-25 lots, spread ~16) = "the wall"
    #  L1 = small noise traders (5-10 lots, 1-2 ticks inside wall)
    #
    #  The wall mid (L2 mid) is the true fair value.
    #  When normalized by wall mid, TOMATOES looks stationary
    #  just like EMERALDS — so we trade it the same way.
    # ══════════════════════════════════════════════════════════

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        product = TOMATOES
        orders: List[Order] = []
        od = state.order_depths[product]
        position = state.position.get(product, 0)

        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])
        asks = sorted(od.sell_orders.items(), key=lambda x: x[0])

        if not bids or not asks:
            return orders

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        # ── Find the wall (largest-volume level) ───────────────
        # Data shows L2 always has bigger volume than L1 (99.2% of ticks)
        # and L1 is always strictly inside L2 (100% of ticks).
        # The wall mid is our fair value.

        wall_bid_info = get_wall_level(bids, "bid")
        wall_ask_info = get_wall_level(asks, "ask")

        if wall_bid_info is None or wall_ask_info is None:
            return orders

        wall_bid = wall_bid_info[0]
        wall_ask = wall_ask_info[0]
        wall_mid = (wall_bid + wall_ask) / 2.0

        # ── PASS 1: TAKING ─────────────────────────────────────
        # Take any order that's mispriced relative to wall mid.
        # Buy below wall_mid (edge > 0), sell above wall_mid (edge > 0).
        # Clear position at 0-EV (at wall_mid) when inventory is large.

        cur_pos = position

        for sp, sv in asks:
            sv = abs(sv)
            buy_room = LIMIT - cur_pos
            if buy_room <= 0:
                break
            if sp < wall_mid:
                # Positive edge: ask is below fair
                vol = min(sv, buy_room)
                orders.append(Order(product, sp, vol))
                cur_pos += vol
            elif sp <= wall_mid and cur_pos < 0:
                # 0-EV clearing when short
                vol = min(sv, abs(cur_pos), buy_room)
                if vol > 0:
                    orders.append(Order(product, sp, vol))
                    cur_pos += vol

        for bp, bv in bids:
            bv = abs(bv)
            sell_room = LIMIT + cur_pos
            if sell_room <= 0:
                break
            if bp > wall_mid:
                # Positive edge: bid is above fair
                vol = min(bv, sell_room)
                orders.append(Order(product, bp, -vol))
                cur_pos -= vol
            elif bp >= wall_mid and cur_pos > 0:
                # 0-EV clearing when long
                vol = min(bv, cur_pos, sell_room)
                if vol > 0:
                    orders.append(Order(product, bp, -vol))
                    cur_pos -= vol

        # ── PASS 2: MAKING ─────────────────────────────────────
        # Place passive orders inside the spread.
        # Penny-jump L1 BBO, but never cross wall_mid.
        # This is identical logic to EMERALDS.

        bid_price = best_bid + 1
        ask_price = best_ask - 1

        # Overbid/undercut logic — same as EMERALDS
        for bp, bv in bids:
            overbid = bp + 1
            if abs(bv) > 1 and overbid < wall_mid:
                bid_price = max(bid_price, overbid)
                break
            elif bp < wall_mid:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in asks:
            undercut = sp - 1
            if abs(sv) > 1 and undercut > wall_mid:
                ask_price = min(ask_price, undercut)
                break
            elif sp > wall_mid:
                ask_price = min(ask_price, sp)
                break

        # Never cross fair (wall mid)
        fair_int = int(round(wall_mid))
        bid_price = min(bid_price, fair_int)
        ask_price = max(ask_price, fair_int)

        # Safety: ensure bid < ask
        if bid_price >= ask_price:
            bid_price = fair_int - 1
            ask_price = fair_int + 1

        remaining_buy = LIMIT - cur_pos
        remaining_sell = LIMIT + cur_pos

        if remaining_buy > 0:
            orders.append(Order(product, bid_price, remaining_buy))
        if remaining_sell > 0:
            orders.append(Order(product, ask_price, -remaining_sell))

        return orders
