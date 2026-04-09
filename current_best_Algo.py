import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple


EMERALDS = "EMERALDS"
TOMATOES = "TOMATOES"

LIMITS = {
    EMERALDS: 80,
    TOMATOES: 80,
}

ADVERSE_EDGE_BUFFER = 0.8
MIN_SIZE_SCALE = 0.35
MAX_SIZE_REDUCTION = 0.65
MIN_LAST_WALL_ABS = 1e-6
MAX_RETURN_ABS = 0.02

DEFAULT_PARAMS = {
    EMERALDS: {
        "fair": 10000.0,
        "take_width": 0.75,
        "clear_width": 0.0,
        "disregard_edge": 0.75,
        "join_edge": 2.0,
        "default_edge": 3.0,
        "soft_pos": 35,
        "adverse_volume": 28,
    },
    TOMATOES: {
        "take_width": 0.6,
        "clear_width": 0.0,
        "disregard_edge": 0.6,
        "join_edge": 1.5,
        "default_edge": 2.0,
        "soft_pos": 40,
        "adverse_volume": 30,
        "ema_alpha": 0.40,
        "mean_reversion_coefficient": -0.12,
    },
}


class Trader:
    def __init__(self, params: Optional[Dict] = None):
        self.params = DEFAULT_PARAMS if params is None else params

    # ---------------------------
    # Generic utilities
    # ---------------------------
    @staticmethod
    def _sorted_book(order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        return bids, asks

    @staticmethod
    def _largest_wall(levels: List[Tuple[int, int]], side: str) -> Optional[int]:
        if not levels:
            return None

        # Pick the level with largest absolute volume.
        # Tie-break: choose more aggressive level (higher for bid, lower for ask).
        best_px, best_vol = levels[0][0], abs(levels[0][1])
        for px, vol in levels:
            av = abs(vol)
            if av > best_vol:
                best_px, best_vol = px, av
            elif av == best_vol:
                if side == "bid" and px > best_px:
                    best_px = px
                elif side == "ask" and px < best_px:
                    best_px = px
        return best_px

    @staticmethod
    def _book_mid(bids: List[Tuple[int, int]], asks: List[Tuple[int, int]]) -> Optional[float]:
        if not bids or not asks:
            return None
        return (bids[0][0] + asks[0][0]) / 2.0

    def _wall_mid(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]]) -> Optional[float]:
        wb = self._largest_wall(bids, "bid")
        wa = self._largest_wall(asks, "ask")
        if wb is None or wa is None:
            return self._book_mid(bids, asks)
        return (wb + wa) / 2.0

    @staticmethod
    def _decode_td(td: str) -> Dict:
        if not td:
            return {}
        try:
            obj = json.loads(td)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _encode_td(obj: Dict) -> str:
        try:
            return json.dumps(obj, separators=(",", ":"))
        except Exception:
            return "{}"

    # ---------------------------
    # Fair-value models
    # ---------------------------
    def _fair_emeralds(self) -> float:
        return self.params[EMERALDS]["fair"]

    def _fair_tomatoes(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]], td: Dict) -> Optional[float]:
        wall_mid = self._wall_mid(bids, asks)
        if wall_mid is None:
            return None

        p = self.params[TOMATOES]
        last_wall = td.get("tom_last_wall_mid")
        ema = td.get("tom_ema", wall_mid)

        # EWM smoothing on wall-mid.
        ema = p["ema_alpha"] * wall_mid + (1.0 - p["ema_alpha"]) * ema

        # Tiny mean-reversion predictor on wall-mid returns.
        if isinstance(last_wall, (int, float)) and abs(last_wall) > MIN_LAST_WALL_ABS:
            ret = (wall_mid - last_wall) / last_wall
            ret = min(MAX_RETURN_ABS, max(-MAX_RETURN_ABS, ret))
            reversion_adjustment = p["mean_reversion_coefficient"] * ret
            fair = ema * (1.0 + reversion_adjustment)
        else:
            fair = ema

        td["tom_last_wall_mid"] = wall_mid
        td["tom_ema"] = ema
        return fair

    # ---------------------------
    # Execution modules
    # ---------------------------
    def _take_orders(
        self,
        product: str,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
        fair: float,
        position: int,
        buy_done: int,
        sell_done: int,
        take_width: float,
        adverse_volume: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        limit = LIMITS[product]

        # Buy mispriced asks
        for ap, av_signed in asks:
            av = abs(av_signed)
            buy_room = limit - (position + buy_done - sell_done)
            if buy_room <= 0:
                break

            edge = fair - ap
            if edge < take_width:
                break

            # Adverse-flow guard: very large displayed volume near fair can be toxic.
            if av > adverse_volume and edge < (take_width + ADVERSE_EDGE_BUFFER):
                continue

            qty = min(av, buy_room)
            if qty > 0:
                orders.append(Order(product, ap, qty))
                buy_done += qty

        # Sell mispriced bids
        for bp, bv_signed in bids:
            bv = abs(bv_signed)
            sell_room = limit + (position + buy_done - sell_done)
            if sell_room <= 0:
                break

            edge = bp - fair
            if edge < take_width:
                break

            if bv > adverse_volume and edge < (take_width + ADVERSE_EDGE_BUFFER):
                continue

            qty = min(bv, sell_room)
            if qty > 0:
                orders.append(Order(product, bp, -qty))
                sell_done += qty

        return orders, buy_done, sell_done

    def _clear_orders(
        self,
        product: str,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
        fair: float,
        position: int,
        buy_done: int,
        sell_done: int,
        clear_width: float,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        limit = LIMITS[product]

        pos_after_take = position + buy_done - sell_done
        buy_room = limit - pos_after_take
        sell_room = limit + pos_after_take

        # Neutralize long inventory at/above fair+width.
        if pos_after_take > 0 and sell_room > 0:
            clear_px = math.ceil(fair + clear_width)
            avail = sum(abs(v) for p, v in bids if p >= clear_px)
            qty = min(pos_after_take, sell_room, avail)
            if qty > 0:
                orders.append(Order(product, clear_px, -qty))
                sell_done += qty
                pos_after_take -= qty

        # Neutralize short inventory at/below fair-width.
        if pos_after_take < 0 and buy_room > 0:
            clear_px = math.floor(fair - clear_width)
            avail = sum(abs(v) for p, v in asks if p <= clear_px)
            qty = min(abs(pos_after_take), buy_room, avail)
            if qty > 0:
                orders.append(Order(product, clear_px, qty))
                buy_done += qty

        return orders, buy_done, sell_done

    def _make_orders(
        self,
        product: str,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
        fair: float,
        position: int,
        buy_done: int,
        sell_done: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        soft_pos: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        limit = LIMITS[product]
        cur_pos = position + buy_done - sell_done

        asks_above = [p for p, _ in asks if p > fair + disregard_edge]
        bids_below = [p for p, _ in bids if p < fair - disregard_edge]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        ask_px = int(round(fair + default_edge))
        bid_px = int(round(fair - default_edge))

        if best_ask_above is not None:
            if abs(best_ask_above - fair) <= join_edge:
                ask_px = best_ask_above
            else:
                ask_px = best_ask_above - 1

        if best_bid_below is not None:
            if abs(fair - best_bid_below) <= join_edge:
                bid_px = best_bid_below
            else:
                bid_px = best_bid_below + 1

        # Inventory skew (stronger near soft limit).
        if cur_pos > soft_pos:
            ask_px -= 1
        elif cur_pos < -soft_pos:
            bid_px += 1

        # Directional fair clamp to avoid accidental crossing around half ticks.
        bid_cap = math.floor(fair)
        ask_floor = math.ceil(fair)
        bid_px = min(bid_px, bid_cap)
        ask_px = max(ask_px, ask_floor)

        if bid_px >= ask_px:
            bid_px = bid_cap - 1
            ask_px = ask_floor + 1
            if bid_px >= ask_px:
                ask_px = bid_px + 1

        buy_room = limit - cur_pos
        sell_room = limit + cur_pos

        # Size modulation: reduce passive size as inventory stretches.
        limit_f = float(limit)
        inv_pressure = min(1.0, abs(cur_pos) / limit_f)
        size_scale = max(MIN_SIZE_SCALE, 1.0 - MAX_SIZE_REDUCTION * inv_pressure)

        buy_qty = max(0, int(buy_room * size_scale))
        sell_qty = max(0, int(sell_room * size_scale))

        if buy_qty > 0:
            orders.append(Order(product, bid_px, buy_qty))
            buy_done += buy_qty
        if sell_qty > 0:
            orders.append(Order(product, ask_px, -sell_qty))
            sell_done += sell_qty

        return orders, buy_done, sell_done

    def _trade_product(
        self,
        product: str,
        order_depth: OrderDepth,
        fair: float,
        position: int,
    ) -> List[Order]:
        bids, asks = self._sorted_book(order_depth)
        if not bids or not asks or fair is None:
            return []

        p = self.params[product]
        buy_done = 0
        sell_done = 0

        take_orders, buy_done, sell_done = self._take_orders(
            product=product,
            bids=bids,
            asks=asks,
            fair=fair,
            position=position,
            buy_done=buy_done,
            sell_done=sell_done,
            take_width=p["take_width"],
            adverse_volume=p["adverse_volume"],
        )

        clear_orders, buy_done, sell_done = self._clear_orders(
            product=product,
            bids=bids,
            asks=asks,
            fair=fair,
            position=position,
            buy_done=buy_done,
            sell_done=sell_done,
            clear_width=p["clear_width"],
        )

        make_orders, _, _ = self._make_orders(
            product=product,
            bids=bids,
            asks=asks,
            fair=fair,
            position=position,
            buy_done=buy_done,
            sell_done=sell_done,
            disregard_edge=p["disregard_edge"],
            join_edge=p["join_edge"],
            default_edge=p["default_edge"],
            soft_pos=p["soft_pos"],
        )

        return take_orders + clear_orders + make_orders

    def run(self, state: TradingState):
        td = self._decode_td(state.traderData)
        result: Dict[str, List[Order]] = {}

        if EMERALDS in state.order_depths:
            fair_e = self._fair_emeralds()
            pos_e = state.position.get(EMERALDS, 0)
            result[EMERALDS] = self._trade_product(
                product=EMERALDS,
                order_depth=state.order_depths[EMERALDS],
                fair=fair_e,
                position=pos_e,
            )

        if TOMATOES in state.order_depths:
            bids_t, asks_t = self._sorted_book(state.order_depths[TOMATOES])
            fair_t = self._fair_tomatoes(bids_t, asks_t, td)
            pos_t = state.position.get(TOMATOES, 0)
            result[TOMATOES] = self._trade_product(
                product=TOMATOES,
                order_depth=state.order_depths[TOMATOES],
                fair=fair_t,
                position=pos_t,
            )

        trader_data = self._encode_td(td)
        conversions = 0
        return result, conversions, trader_data
