"""Delivery Promise Monitor. Run with: streamlit run app.py"""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


TEAM_NAME = "team 10 of fin"
DATA_FILE = Path(__file__).with_name("olist_orders_dataset.csv")
SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
PLANNING_TARGET = 95.0  # Proposed target for this project, not Olist's official SLA.
DATE_COLUMNS = [
    "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
    "order_delivered_customer_date", "order_estimated_delivery_date",
]
BAND_LABELS = ["0–3", "3–7", "7–14", "14–21", "21–30", ">30"]
TEAL, ORANGE = "#147d83", "#ba542d"


def prepare_orders(raw):
    required = {"order_id", "customer_id", "order_status", *DATE_COLUMNS}
    if not required.issubset(raw.columns):
        raise ValueError("Missing columns: " + ", ".join(sorted(required - set(raw.columns))))
    if raw.order_id.isna().any() or raw.order_id.duplicated().any():
        raise ValueError("Order IDs must be present and unique. Check the source CSV.")
    orders = raw.copy()
    parse_failures = {}
    for column in DATE_COLUMNS:
        orders[column] = pd.to_datetime(raw[column], errors="coerce")
        parse_failures[column] = int((raw[column].notna() & orders[column].isna()).sum())
    orders["delivery_days"] = (
        orders.order_delivered_customer_date - orders.order_purchase_timestamp
    ).dt.total_seconds() / 86400
    orders["late_days"] = (
        orders.order_delivered_customer_date.dt.normalize()
        - orders.order_estimated_delivery_date.dt.normalize()
    ).dt.days
    orders["eligible"] = (
        orders.order_status.eq("delivered") & orders.order_purchase_timestamp.notna()
        & orders.order_delivered_customer_date.notna()
        & orders.order_estimated_delivery_date.notna() & orders.delivery_days.ge(0)
    )
    orders["on_time"] = orders.late_days.le(0)
    return orders, parse_failures


@st.cache_data(show_spinner="Preparing delivery records...")
def load_orders(path, modified_ns):
    # modified_ns is part of the cache key, so replacing the file reloads data.
    return prepare_orders(pd.read_csv(path, dtype={"order_id": "str", "customer_id": "str"}))


def select_deliveries(orders, start, end):
    """Filter by actual delivery date, including all times on the end date."""
    return orders.loc[
        orders.eligible
        & orders.order_delivered_customer_date.ge(pd.Timestamp(start))
        & orders.order_delivered_customer_date.lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    ].copy()


def weekly_summary(deliveries, start, end):
    week = deliveries.order_delivered_customer_date.dt.to_period("W-SUN").dt.start_time
    weekly = deliveries.groupby(week).agg(
        delivered_orders=("order_id", "size"), on_time_orders=("on_time", "sum")
    )
    first = pd.Timestamp(start).to_period("W-SUN").start_time
    last = pd.Timestamp(end).to_period("W-SUN").start_time
    weekly = weekly.reindex(pd.date_range(first, last, freq="W-MON"))
    for column in ["delivered_orders", "on_time_orders"]:
        weekly[column] = weekly[column].fillna(0).astype(int)
    weekly["on_time_pct"] = (
        weekly.on_time_orders / weekly.delivered_orders.where(weekly.delivered_orders.gt(0)) * 100
    )
    weekly["late_orders"] = weekly.delivered_orders - weekly.on_time_orders
    weekly.index.name = "Week starting Monday"
    return weekly


def duration_summary(deliveries):
    view = deliveries.copy()
    view["band"] = pd.cut(
        view.delivery_days, bins=[0, 3, 7, 14, 21, 30, float("inf")],
        labels=BAND_LABELS, include_lowest=True, right=True,
    )
    view["result"] = view.on_time.map({True: "On time", False: "Late"})
    counts = view.groupby(["band", "result"], observed=False).size().unstack(fill_value=0)
    return counts.reindex(index=BAND_LABELS, columns=["On time", "Late"], fill_value=0)


def weekly_figure(weekly):
    figure = go.Figure(go.Scatter(
        x=weekly.index, y=weekly.on_time_pct, mode="lines+markers", connectgaps=False,
        line={"color": TEAL, "width": 3}, marker={"size": 7},
        customdata=weekly[["delivered_orders", "on_time_orders", "late_orders"]].values,
        hovertemplate=("Week starting %{x|%b %d, %Y}<br>On time: %{y:.2f}%"
                       "<br>Delivered: %{customdata[0]:,}<br>On time: %{customdata[1]:,}"
                       "<br>Late: %{customdata[2]:,}<extra></extra>"),
        name="On-time rate",
    ))
    figure.add_hline(
        y=PLANNING_TARGET, line_dash="dash", line_color=ORANGE,
        annotation_text=f"{PLANNING_TARGET:.0f}% planning target", annotation_position="bottom right",
    )
    figure.update_layout(
        height=350, margin={"l": 0, "r": 15, "t": 10, "b": 0},
        xaxis_title="Actual delivery week", yaxis_title="On-time deliveries (%)",
        yaxis={"range": [0, 100], "ticksuffix": "%"}, showlegend=False,
    )
    return figure


def duration_figure(counts):
    figure = go.Figure()
    for name, color in [("On time", TEAL), ("Late", ORANGE)]:
        figure.add_bar(
            name=name, x=BAND_LABELS, y=counts[name], marker_color=color,
            marker_pattern_shape="/" if name == "Late" else "",
            hovertemplate=f"Delivery time: %{{x}} days<br>{name}: %{{y:,}} orders<extra></extra>",
        )
    figure.update_layout(
        barmode="stack", height=350, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="Purchase-to-delivery time (days)", yaxis_title="Delivered orders",
        legend={"orientation": "h", "y": 1.14, "x": 0},
        xaxis={"categoryorder": "array", "categoryarray": BAND_LABELS},
    )
    return figure


def main():
    st.set_page_config(page_title="Delivery Promise Monitor", layout="wide")
    st.title(f"Delivery Promise Monitor — {TEAM_NAME}")
    st.caption(
        "For the delivery operations manager to identify missed delivery promises "
        "and decide when to investigate fulfillment capacity and delivery commitments."
    )
    try:
        orders, parse_failures = load_orders(str(DATA_FILE), DATA_FILE.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Cannot load the order file: {error}")
        st.info("Keep olist_orders_dataset.csv in the same folder as app.py.")
        st.stop()
    eligible = orders.loc[orders.eligible]
    if eligible.empty:
        st.warning("No delivered orders have usable purchase, actual and estimated delivery dates.")
        st.stop()
    minimum = eligible.order_delivered_customer_date.min().date()
    maximum = eligible.order_delivered_customer_date.max().date()
    purchase = orders.order_purchase_timestamp.dropna()
    st.caption(
        f"Data: purchases {purchase.min():%b %d, %Y}–{purchase.max():%b %d, %Y}; "
        f"observed deliveries {minimum:%b %d, %Y}–{maximum:%b %d, %Y}. Kaggle snapshot. "
        "Proposed production refresh: nightly; manager review: every morning."
    )
    st.sidebar.header("Review period")
    selected = st.sidebar.date_input(
        "Actual delivery date range",
        value=(min(max(date(2018, 6, 1), minimum), maximum), min(max(date(2018, 8, 31), minimum), maximum)),
        min_value=minimum, max_value=maximum, format="YYYY-MM-DD",
    )
    st.sidebar.caption("Default: Jun–Aug 2018. This filter uses completed delivery dates, not purchase dates.")
    st.sidebar.markdown(f"[Olist dataset on Kaggle]({SOURCE_URL})")
    st.sidebar.caption(f"Planning target: {PLANNING_TARGET:.0f}% on time. Project assumption; not an official Olist SLA.")
    if not isinstance(selected, (tuple, list)) or len(selected) != 2:
        st.info("Choose both a start date and an end date.")
        st.stop()
    start, end = selected
    if start > end:
        st.warning("Choose an end date on or after the start date.")
        st.stop()
    deliveries = select_deliveries(orders, start, end)
    n = len(deliveries)
    on_time = int(deliveries.on_time.sum())
    late_count = n - on_time
    rate = on_time / n * 100 if n else None
    p90 = deliveries.delivery_days.quantile(0.9) if n else None
    left, middle, right = st.columns(3)
    left.metric("Delivered Orders", f"{n:,}", border=True)
    middle.metric("On-Time Delivery Rate", f"{rate:.2f}%" if n else "N/A", border=True)
    right.metric(
        "90th Percentile Delivery Time", f"{p90:.2f} days" if n else "N/A", border=True,
        help="90% of eligible completed deliveries took this long or less from purchase. Days are continuous 24-hour periods.",
    )
    st.caption(
        f"Completed-delivery window: {start:%Y-%m-%d}–{end:%Y-%m-%d}. "
        f"On time: {on_time:,}; late: {late_count:,}. On time means actual calendar date ≤ promised calendar date. "
        "All three metrics use the same eligible delivered orders."
    )
    if not n:
        st.info("No eligible deliveries in this period. Select another date range; unavailable rates are not zero.")
        st.stop()

    if n < 100:
        st.info("Fewer than 100 deliveries are selected. Treat percentages and tail estimates as a small-sample review.")
    elif rate < PLANNING_TARGET:
        st.warning(f"Selected-period on-time performance is {PLANNING_TARGET - rate:.2f} percentage points below the proposed target. Review late orders before deciding on corrective action.")
    else:
        st.info(f"Selected-period performance meets the proposed {PLANNING_TARGET:.0f}% benchmark. {late_count:,} late deliveries still need proportionate review.")

    weekly = weekly_summary(deliveries, start, end)
    counts = duration_summary(deliveries)
    chart_left, chart_right = st.columns([1.2, 1])
    with chart_left:
        st.subheader("Is promise reliability changing?")
        st.plotly_chart(weekly_figure(weekly), width="stretch", key="weekly_reliability",
                        config={"displaylogo": False, "scrollZoom": False})
        st.caption("Weeks start Monday. Hover for sample sizes. Empty weeks stay as gaps; edge weeks include only selected dates.")
    with chart_right:
        st.subheader("Where are the long waits?")
        st.plotly_chart(duration_figure(counts), width="stretch", key="delivery_duration",
                        config={"displaylogo": False, "scrollZoom": False})
        st.caption("Intervals include the upper boundary: 3–7 means >3 and ≤7 days. All long waits are retained in >30. A long wait can still meet a generous promise.")

    with st.expander(f"Investigate late deliveries · {late_count:,} orders"):
        details = deliveries.loc[~deliveries.on_time, [
            "order_id", "order_purchase_timestamp", "order_estimated_delivery_date",
            "order_delivered_customer_date", "late_days", "delivery_days",
        ]].sort_values(["late_days", "delivery_days"], ascending=False)
        if details.empty:
            st.write("No late deliveries in this period.")
        else:
            st.dataframe(details.head(300), hide_index=True, width="stretch")
            st.caption("Showing the 300 longest delays. Download includes every selected late delivery.")
            st.download_button("Download late deliveries", details.to_csv(index=False).encode("utf-8"),
                               file_name=f"late_deliveries_{start}_{end}.csv", mime="text/csv")
    with st.expander("Weekly counts and metric definitions"):
        st.dataframe(weekly.reset_index(), hide_index=True, width="stretch")
        st.write("Delivery-duration counts (the data behind the stacked bars):")
        st.dataframe(counts.rename_axis("Delivery days").reset_index(), hide_index=True, width="stretch")
        st.markdown(
            "**Delivered Orders:** orders with delivered status, complete purchase/actual/estimated dates and nonnegative total duration. "
            "**On-Time Rate:** on-time eligible deliveries divided by all eligible deliveries. "
            "**P90:** the 90th percentile of purchase-to-delivery time, using linear interpolation. "
            "The dashed 95% line is a planning assumption, not a measured contract requirement."
        )
    with st.expander("Data quality, scope and refresh policy"):
        delivered_status = orders.order_status.eq("delivered")
        missing_actual = int((delivered_status & orders.order_delivered_customer_date.isna()).sum())
        inconsistent = int((~delivered_status & orders.order_delivered_customer_date.notna()).sum())
        st.write(
            f"Source: {len(orders):,} orders. Eligible completed deliveries: {len(eligible):,}. "
            f"{missing_actual:,} orders have delivered status without an actual delivery date; "
            f"{inconsistent:,} have another status despite an actual delivery date. These are excluded. "
            f"Nonempty timestamps that failed parsing: {sum(parse_failures.values()):,}."
        )
        st.dataframe(orders[DATE_COLUMNS].isna().sum().rename("Missing values").to_frame(), width="stretch")
        st.write(
            "This is a fixed historical snapshot. Recent purchases and unfinished orders are not represented in completed-delivery rates. "
            "The file has no extraction timestamp or status history, so it cannot measure today's overdue backlog. "
            "Sparse final weeks do not establish a demand collapse. Timestamps have no explicit timezone; no conversion is assumed. "
            "Revenue, margin, locations, carrier identity and repeat-customer identity are absent from this table."
        )
        st.write(
            "Proposed production policy: sync completed delivery events nightly, review each morning, and review capacity each Monday. "
            "An initial investigation rule is a trailing 7-day on-time rate below 95%, with at least 100 deliveries, "
            "on two successive daily checks. This needs calibration and is not a live alert in this historical app. "
            "A day of stale data delays investigation; a week can hide a persistent problem. "
            "The source has no complaint or compensation costs, so no monetary savings are claimed."
        )


if __name__ == "__main__":
    main()
