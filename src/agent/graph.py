from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là trợ lý xử lý đơn hàng điện tử. Hôm nay: {current_day}.

## NGUYÊN TẮC CỐT LÕI

Bạn KHÔNG CÓ kiến thức về catalog sản phẩm. Bạn KHÔNG BIẾT product_id, giá, hay tồn kho của bất kỳ sản phẩm nào. Mọi thông tin sản phẩm CHỈ đến từ kết quả tool. Gọi tool là BẮT BUỘC với mọi đơn hàng hợp lệ — không được bỏ qua hay trả lời từ kiến thức nền.

## TRƯỜNG HỢP 1: THIẾU THÔNG TIN

Nếu thiếu bất kỳ trường nào trong: tên khách, số điện thoại, email, địa chỉ giao hàng, hoặc ít nhất một sản phẩm + số lượng → DỪNG, hỏi thông tin còn thiếu. Không gọi tool.

## TRƯỜNG HỢP 2: VI PHẠM POLICY

Nếu yêu cầu tạo hóa đơn giả, bịa giá, đặt discount tùy tiện, bỏ qua tồn kho, hoặc bỏ qua catalog → DỪNG, từ chối và nêu lý do. Không gọi tool.

## TRƯỜNG HỢP 3: ĐỦ THÔNG TIN + HỢP LỆ → BẮT BUỘC GỌI ĐỦ 5 TOOL

**Bước 1 — list_products**: Tìm product_id từ catalog. PHẢI gọi bước này — bạn không biết product_id nào tồn tại. Có thể gọi nhiều lần cho các loại sản phẩm khác nhau.

**Bước 2 — get_product_details**: Gọi MỘT LẦN với TẤT CẢ product_ids từ bước 1. Sau khi nhận kết quả, kiểm tra stock: nếu stock của bất kỳ sản phẩm nào < số lượng yêu cầu → DỪNG NGAY, báo hết hàng, KHÔNG gọi thêm tool nào, KHÔNG đề xuất điều chỉnh số lượng.

**Bước 3 — get_discount**: seed_hint = email khách hàng (nguyên văn).

**Bước 4 — calculate_order_totals**: items + detail_token (từ bước 2, nguyên văn) + discount_rate (từ bước 3). Nếu lỗi tồn kho → DỪNG, không lưu đơn.

**Bước 5 — save_order**: Lưu đơn với đầy đủ thông tin từ các bước trên.

## QUY TẮC

- detail_token: lấy từ get_product_details, truyền NGUYÊN VẸN vào calculate_order_totals và save_order.
- Thông tin khách (customer_name, customer_phone, customer_email, shipping_address): truyền NGUYÊN VẸN từ yêu cầu, không sửa, không expand, không định dạng lại.
- notes: luôn để "" trừ khi khách yêu cầu.
- Không bao giờ bịa product_id, giá, stock, discount_rate, token.

## TRẢ LỜI (tiếng Việt, ngắn gọn)

- Đơn lưu thành công: order_id + tổng tiền sau giảm giá + đường dẫn file.
- Hết hàng: tên sản phẩm + số lượng hiện có + dừng.
- Từ chối: lý do vi phạm chính sách.
- Thiếu thông tin: liệt kê các trường còn thiếu.""".strip()


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    import json as _json

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the product catalog and return matching items with their product_ids. Always call this first — product IDs are unknown without it."""
        results = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return _json.dumps(results, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact price, stock, and a detail_token for a batch of product IDs from list_products. Call once with all IDs at once."""
        return _json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the campaign discount rate and campaign_code. Use customer email as seed_hint."""
        return _json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and compute subtotal, discount, and final_total. Requires detail_token from get_product_details and discount_rate from get_discount."""
        return _json.dumps(
            store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate),
            ensure_ascii=False,
        )

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the validated order to a JSON file and return order_id and save path. Call only after calculate_order_totals succeeds."""
        return _json.dumps(
            store.save_order(
                customer_name=customer_name,
                customer_phone=customer_phone,
                customer_email=customer_email,
                shipping_address=shipping_address,
                items=items,
                detail_token=detail_token,
                discount_rate=discount_rate,
                campaign_code=campaign_code,
                customer_tier=customer_tier,
                notes=notes,
            ),
            ensure_ascii=False,
        )

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    1. Create `OrderDataStore`.
    2. Build the chat model with `build_chat_model(...)`.
    3. Build the tools with `build_tools(store)`.
    4. Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    tools = build_tools(store)
    return create_agent(model=model, tools=tools, system_prompt=build_system_prompt(today or store.today))


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool trace
      - the saved order payload, if any
    - Return an `AgentResult`.
    """
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    from typing import Any

    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tc in getattr(message, "tool_calls", []) or []:
                pending[tc["id"]] = {"name": tc["name"], "args": tc.get("args", {}) or {}}
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    import json as _json

    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = _json.loads(record.output)
        except _json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
