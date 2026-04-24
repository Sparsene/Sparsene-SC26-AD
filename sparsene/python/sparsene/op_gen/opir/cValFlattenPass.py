from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sympy import Integer, Rational, Symbol, simplify
from sympy import Number as SymNumber
from sympy import sympify

from sparsene.op_gen.opir.op_ir import (
    Op,
    OpOperand,
    ConstantOp,
    ForLoopOp,
    ArrayRefOp,
    Value,
    IntType,
    MetaOp,
    ExternalSymbolOp,
    AddOp,
    MulOp,
    DivOp,
    PowOp,
    ArrayType,
    OpResult,
)
from sparsene.op_gen.opir.ops import (
    CValLoadOp,
    CValStoreOp,
)


class CValFlattenPass:
    """
        C_val   ：
    1)     C_val        [..., N]     [M, N]
    2)   C    /              (offset) + len
    3)            C     
    """

    def __init__(
        self, 
        op_builder=None, 
        varlen2IdxArrayTable: Optional[Dict[str, Any]] = None
    ):
        self.op_builder = op_builder
        #          ，   {'varlen(M)': ArrayDef(val_sidx, ...)}
        self.varlen2IdxArrayTable = varlen2IdxArrayTable or {}
        
        self._root_c_val: Optional[Value] = None
        self._root_c_type: Optional[ArrayType] = None
        self._c_alias_offsets: Dict[Value, Optional[Value]] = {}
        self._m_sparse_row_sidx: Optional[Value] = None
        print("Initialized CValFlattenPass with varlen2IdxArrayTable:", self.varlen2IdxArrayTable)

    def run(self, ops):
        root = self._normalize_root(ops)
        if root is None:
            return ops

        c_val_ext = self._find_c_val_external(root)
        if not c_val_ext:
            return root

        assert isinstance(c_val_ext.result.type, ArrayType), "C_val should be an array"
        old_c_type = c_val_ext.result.type
        if len(old_c_type.dims) < 2:
            return root
            
        #    BLK_M    ，     1D sidx       
        self._c_blk_m_expr = old_c_type.dims[-2] if len(old_c_type.dims) >= 2 else Integer(1)
        self._m_sparse_row_sidx = self._find_m_sparse_row_sidx(root, old_c_type)
            
        c_outer_dim = str(old_c_type.dims[0])
        is_sparse_m = self._is_varlen_m_dim(c_outer_dim) or self._m_sparse_row_sidx is not None

        #    C        ，                [M, N]
        if is_sparse_m:
            flat_m = Symbol("M")
        else:
            flat_m = self._product_expr(old_c_type.dims[:-1])
            
        flat_n = old_c_type.dims[-1]
        self._root_c_type = ArrayType([flat_m, flat_n], old_c_type.datatype)
        c_val_ext.result.type = self._root_c_type
        self._root_c_val = c_val_ext.result

        self._c_alias_offsets = {self._root_c_val: None}

        self._rewrite_block(root.block)
        return root

    def _rewrite_block(
        self,
        block,
        incoming_replace_map: Optional[Dict[Value, Value]] = None,
    ):
        replace_map = dict(incoming_replace_map or {})
        new_ops = []

        for op in block.ops:
            self._remap_op_operands(op, replace_map)
            prefix_ops: List[Op] = []

            if isinstance(op, ForLoopOp):
                self._propagate_c_alias_for_loop(op)
                self._rewrite_block(op.body, replace_map)
                new_ops.extend(prefix_ops)
                new_ops.append(op)
                continue

            if (
                isinstance(op, ArrayRefOp)
                and len(op.operands) >= 2
                and self._is_c_alias(op.operands[0].source)
            ):
                rewritten_load = self._rewrite_c_array_ref(op, prefix_ops)
                replace_map[op.result] = rewritten_load.result
                new_ops.extend(prefix_ops)
                new_ops.append(rewritten_load)
                continue

            if (
                isinstance(op, CValLoadOp)
                and len(op.operands) >= 2
                and self._is_c_alias(op.operands[0].source)
            ):
                self._rewrite_c_val_load(op, prefix_ops)
                new_ops.extend(prefix_ops)
                new_ops.append(op)
                continue

            if (
                isinstance(op, CValStoreOp)
                and len(op.operands) >= 3
                and self._is_c_alias(op.operands[1].source)
            ):
                self._rewrite_c_val_store(op, prefix_ops)
                new_ops.extend(prefix_ops)
                new_ops.append(op)
                continue

            new_ops.extend(prefix_ops)
            new_ops.append(op)

        block.ops = new_ops

    def _propagate_c_alias_for_loop(self, op: ForLoopOp):
        for i in range(op.num_iter_args):
            incoming = op.operands[i + 2].source
            if not self._is_c_alias(incoming):
                continue

            iter_arg = op.get_iter_arg(i)
            iter_arg.type = incoming.type
            op.results[i].type = incoming.type

            base_offset = self._c_alias_offsets[incoming]
            self._c_alias_offsets[iter_arg] = base_offset
            self._c_alias_offsets[op.results[i]] = base_offset

    def _rewrite_c_array_ref(self, op: ArrayRefOp, generated_ops: List[Op]) -> CValLoadOp:
        assert self._root_c_val is not None
        assert isinstance(op.result.type, ArrayType)

        parent_alias = op.operands[0].source
        logical_idx = op.operands[1].source
        length_val = self._infer_length_from_tile_type(op.result.type, generated_ops)

        sparse_row_indices = self._build_sparse_row_indices(parent_alias, logical_idx, generated_ops)
        if sparse_row_indices is not None:
            load_op = CValLoadOp(
                mem="S2R",
                array=self._root_c_val,
                offset=sparse_row_indices,
                length=length_val,
            )
            load_result = OpResult(
                type=op.result.type,
                defining_op=load_op,
                result_idx_in_owner=0,
                name_hint=op.result.name_hint,
            )
            load_op.add_result(load_result)
            load_op.set_attribute("c_sparse_indexed", True)
            return load_op

        abs_offset = self._build_absolute_offset(
            parent_alias,
            logical_idx,
            length_val,
            generated_ops,
        )

        load_op = CValLoadOp(
            mem="S2R",
            array=self._root_c_val,
            offset=abs_offset,
            length=length_val,
        )
        load_result = OpResult(
            type=op.result.type,
            defining_op=load_op,
            result_idx_in_owner=0,
            name_hint=op.result.name_hint,
        )
        load_op.add_result(load_result)

        self._c_alias_offsets[load_result] = abs_offset
        return load_op

    def _rewrite_c_val_load(self, op: CValLoadOp, generated_ops: List[Op]):
        assert self._root_c_val is not None

        parent_alias = op.operands[0].source
        logical_idx = op.operands[1].source
        length_val = self._ensure_len_value(op, generated_ops)

        sparse_row_indices = self._build_sparse_row_indices(parent_alias, logical_idx, generated_ops)
        if sparse_row_indices is not None:
            op.operands = [OpOperand(self._root_c_val), OpOperand(sparse_row_indices)]

            if op.num_results == 0:
                inferred_type = self._infer_load_result_type(length_val)
                load_result = OpResult(
                    type=inferred_type,
                    defining_op=op,
                    result_idx_in_owner=0,
                    name_hint="c_slice",
                )
                op.add_result(load_result)

            op.set_attribute("c_flattened", True)
            op.set_attribute("c_sparse_indexed", True)
            return

        abs_offset = self._build_absolute_offset(
            parent_alias,
            logical_idx,
            length_val,
            generated_ops,
        )

        op.operands = [OpOperand(self._root_c_val), OpOperand(abs_offset)]

        if op.num_results == 0:
            inferred_type = self._infer_load_result_type(length_val)
            load_result = OpResult(
                type=inferred_type,
                defining_op=op,
                result_idx_in_owner=0,
                name_hint="c_slice",
            )
            op.add_result(load_result)

        self._c_alias_offsets[op.result] = abs_offset
        op.set_attribute("c_flattened", True)

    def _rewrite_c_val_store(self, op: CValStoreOp, generated_ops: List[Op]):
        assert self._root_c_val is not None

        to_store = op.operands[0].source
        parent_alias = op.operands[1].source
        logical_idx = op.operands[2].source
        length_val = self._ensure_len_value(op, generated_ops)

        sparse_row_indices = self._build_sparse_row_indices(parent_alias, logical_idx, generated_ops)
        if sparse_row_indices is not None:
            op.operands = [
                OpOperand(to_store),
                OpOperand(self._root_c_val),
                OpOperand(sparse_row_indices),
            ]
            op.set_attribute("c_flattened", True)
            op.set_attribute("c_sparse_indexed", True)
            return

        abs_offset = self._build_absolute_offset(
            parent_alias,
            logical_idx,
            length_val,
            generated_ops,
        )

        op.operands = [
            OpOperand(to_store),
            OpOperand(self._root_c_val),
            OpOperand(abs_offset),
        ]
        op.set_attribute("c_flattened", True)

    def _build_absolute_offset(
        self,
        parent_alias: Value,
        logical_idx: Value,
        length_val: Value,
        generated_ops: List[Op],
    ) -> Value:
        mul = MulOp(logical_idx, length_val)
        generated_ops.append(mul)
        local_offset = mul.result

        base_offset = self._c_alias_offsets.get(parent_alias)
        if base_offset is None:
            return local_offset

        add = AddOp(base_offset, local_offset)
        generated_ops.append(add)
        return add.result

    def _ensure_len_value(self, op: Op, generated_ops: List[Op]) -> Value:
        len_member = getattr(op, "len", None)
        if isinstance(len_member, Value):
            return len_member

        if isinstance(op, CValLoadOp) and op.num_results > 0 and isinstance(op.result.type, ArrayType):
            inferred = self._infer_length_from_tile_type(op.result.type, generated_ops)
            op.len = inferred
            return inferred

        if isinstance(op, CValStoreOp) and isinstance(op.operands[0].source.type, ArrayType):
            inferred = self._infer_length_from_tile_type(op.operands[0].source.type, generated_ops)
            op.len = inferred
            return inferred

        one = self._expr_to_value(Integer(1), generated_ops)
        if isinstance(op, (CValLoadOp, CValStoreOp)):
            op.len = one
        return one

    def _infer_length_from_tile_type(self, tile_type: ArrayType, generated_ops: List[Op]) -> Value:
        if len(tile_type.dims) <= 1:
            return self._expr_to_value(Integer(1), generated_ops)
        length_expr = self._product_expr(tile_type.dims[:-1])
        return self._expr_to_value(length_expr, generated_ops)

    def _infer_load_result_type(self, length_val: Value) -> ArrayType:
        assert self._root_c_type is not None
        n_dim = self._root_c_type.dims[-1]
        length_expr = self._value_to_expr(length_val)
        return ArrayType([length_expr, n_dim], self._root_c_type.datatype)

    def _remap_op_operands(self, op: Op, replace_map: Dict[Value, Value]) -> None:
        if not replace_map:
            return

        if op.num_operands > 0:
            remapped_operands: List[OpOperand] = []
            for operand in op.operands:
                source = operand.source
                while source in replace_map:
                    source = replace_map[source]
                remapped_operands.append(OpOperand(source))
            op.operands = remapped_operands

        if hasattr(op, "len") and isinstance(getattr(op, "len"), Value):
            len_source = getattr(op, "len")
            while len_source in replace_map:
                len_source = replace_map[len_source]
            setattr(op, "len", len_source)

    def _is_c_alias(self, value: Value) -> bool:
        return value in self._c_alias_offsets

    def _is_varlen_m_dim(self, dim: Any) -> bool:
        dim_str = str(dim)
        return "varlen(M)" in dim_str or "nnz_dim_M" in dim_str

    def _is_sparse_m_c_container(self, value: Value) -> bool:
        if self._m_sparse_row_sidx is None:
            return False
        if not isinstance(value.type, ArrayType):
            return False
        if len(value.type.dims) < 3:
            return False

        dims = value.type.dims
        if not self._is_varlen_m_dim(dims[0]):
            return False

        sidx_type = self._m_sparse_row_sidx.type
        if not isinstance(sidx_type, ArrayType) or len(sidx_type.dims) < 2:
            return False

        #    C:[varlen(M)/BLK_M][BLK_M][N], sidx:[varlen(M)/BLK_M][BLK_M]
        return str(dims[1]) == str(sidx_type.dims[1])

    def _build_sparse_row_indices(
        self,
        parent_alias: Value,
        logical_idx: Value,
        generated_ops: List[Op],
    ) -> Optional[Value]:
        if self._m_sparse_row_sidx is None:
            return None
        
        sidx_type = self._m_sparse_row_sidx.type
        
        if len(sidx_type.dims) == 1:
            # 1D   : load(idx) * BLK_M 
            from sparsene.op_gen.opir.ops import LoadOp
            load_op = LoadOp(self._m_sparse_row_sidx, [logical_idx])
            generated_ops.append(load_op)
            
            blk_m_val = self._expr_to_value(self._c_blk_m_expr, generated_ops)
            mul_op = MulOp(load_op.result, blk_m_val, name_hint="sparse_row_offset")
            generated_ops.append(mul_op)
            return mul_op.result
        else:
            # 2D   :       BLK_M   idx     
            placeholder = Value(IntType(), name_hint="_")
            
            # ---     ：   name_hint    ---
            row_idx_ref = ArrayRefOp(self._m_sparse_row_sidx, [logical_idx, placeholder])
            
            #      result    
            if row_idx_ref.num_results > 0:
                row_idx_ref.result.name_hint = "sidx_slice"
                
            generated_ops.append(row_idx_ref)
            return row_idx_ref.result

    def _find_m_sparse_row_sidx(self, root: MetaOp, c_type: ArrayType) -> Optional[Value]:
        # 1.          (         )
        if self.varlen2IdxArrayTable and 'varlen(M)' in self.varlen2IdxArrayTable:
            base_name = self.varlen2IdxArrayTable['varlen(M)'].name
            for op in root.block.ops:
                if isinstance(op, ExternalSymbolOp):
                    symbol_name = str(op.attributes.get("symbol", op.result.name_hint or ""))
                    if symbol_name.startswith(base_name):
                        return op.result

        # 2.       ，        (Heuristic Search)
        c_outer_dim = str(c_type.dims[0])
        c_blk_m_dim = c_type.dims[1] if len(c_type.dims) > 1 else None

        for op in root.block.ops:
            if not isinstance(op, ExternalSymbolOp):
                continue
            if not isinstance(op.result.type, ArrayType) or not isinstance(op.result.type.datatype, IntType):
                continue

            sidx_dims = list(op.result.type.dims)
            if len(sidx_dims) == 0:
                continue

            #       1:      sidx   idx
            symbol_name = str(op.attributes.get("symbol", op.result.name_hint or ""))
            if "sidx" not in symbol_name and "idx" not in symbol_name:
                continue

            #       2:       nnz_dim_M   varlen(M)
            first_dim_match = (str(sidx_dims[0]) == c_outer_dim)
            if not first_dim_match:
                first_dim_match = self._is_varlen_m_dim(sidx_dims[0]) and self._is_varlen_m_dim(c_outer_dim)

            if first_dim_match:
                #       3:     2D   ，          BLK_M
                if len(sidx_dims) >= 2 and c_blk_m_dim is not None:
                    if str(sidx_dims[1]) == str(c_blk_m_dim):
                        return op.result
                #     1D   ，    
                elif len(sidx_dims) == 1:
                    return op.result

        return None

    def _find_c_val_external(self, root: MetaOp) -> Optional[ExternalSymbolOp]:
        for op in root.block.ops:
            if not isinstance(op, ExternalSymbolOp):
                continue
            
            #      symbol name
            symbol = str(op.attributes.get("symbol", op.result.name_hint or ""))
            
            #     C_val    (   C_val_new, C_val_1_new  )     
            if symbol.startswith("C_val"):
                return op
                
        return None

    def _product_expr(self, dims: Sequence[Any]) -> Any:
        acc = sympify(1)
        for dim in dims:
            acc = simplify(acc * sympify(dim))
        return acc

    def _value_to_expr(self, value: Value) -> Any:
        if isinstance(value, OpResult) and isinstance(value.defining_op, ConstantOp):
            try:
                return simplify(sympify(value.defining_op.attributes.get("value")))
            except Exception:
                pass
        if value.name_hint is not None:
            return Symbol(value.name_hint)
        return Symbol("len")
    
    def _normalize_root(self, ops) -> Optional[MetaOp]:
        if isinstance(ops, MetaOp):
            return ops
        if isinstance(ops, Sequence) and len(ops) == 1 and isinstance(ops[0], MetaOp):
            return ops[0]
        return None

    def _expr_to_value(self, expr: Any, generated_ops: List[Op]) -> Value:
        expr = simplify(sympify(expr))

        if isinstance(expr, Integer):
            const = ConstantOp(int(expr), IntType())
            generated_ops.append(const)
            return const.result

        if isinstance(expr, Rational):
            num = ConstantOp(int(expr.p), IntType())
            den = ConstantOp(int(expr.q), IntType())
            generated_ops.extend([num, den])
            div = DivOp(num.result, den.result)
            generated_ops.append(div)
            return div.result

        if isinstance(expr, SymNumber):
            const = ConstantOp(int(expr), IntType())
            generated_ops.append(const)
            return const.result

        if isinstance(expr, Symbol):
            name = str(expr)
            external = ExternalSymbolOp(name, IntType())
            external.result.name_hint = name
            generated_ops.append(external)
            return external.result

        if getattr(expr, "is_Add", False):
            args = list(expr.args)
            acc = self._expr_to_value(args[0], generated_ops)
            for arg in args[1:]:
                rhs = self._expr_to_value(arg, generated_ops)
                add = AddOp(acc, rhs)
                generated_ops.append(add)
                acc = add.result
            return acc

        if getattr(expr, "is_Mul", False):
            args = list(expr.args)
            acc = self._expr_to_value(args[0], generated_ops)
            for arg in args[1:]:
                rhs = self._expr_to_value(arg, generated_ops)
                mul = MulOp(acc, rhs)
                generated_ops.append(mul)
                acc = mul.result
            return acc

        if getattr(expr, "is_Pow", False):
            base = self._expr_to_value(expr.args[0], generated_ops)
            exp = self._expr_to_value(expr.args[1], generated_ops)
            pow_op = PowOp(base, exp)
            generated_ops.append(pow_op)
            return pow_op.result

        fallback = ExternalSymbolOp(str(expr), IntType())
        fallback.result.name_hint = str(expr)
        generated_ops.append(fallback)
        return fallback.result