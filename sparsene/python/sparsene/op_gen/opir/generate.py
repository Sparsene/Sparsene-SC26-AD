from __future__ import annotations

from sparsene.op_gen.computent.computent import computent_from_rts, Computent, ArrayDef
from sparsene.format.format import Format, Expr, Symbol, Number
from sparsene.op_gen.computent.computent import ArrayDefCollection
from typing import Any, List, Optional, Dict, Literal, Tuple, Sequence
from sparsene.op_gen.computent.computent import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
    ArrayRef,
    DataType,
)
from dataclasses import dataclass
from sparsene.op_gen.opir.op_ir import (
    Op,
    DeviceOp,
    OpBuilder,
    SymbolTable,
    ConstantOp,
    ForLoopOp,
    ArrayRefOp,
    LoadOp,
    Value,
    IntType,
    MetaOp,
    ExternalSymbolOp,
    AddOp,
    MulOp,
    DivOp,
    PowOp,
    ArangeOp,
    op_builder,
    Type,
    ArrayType,
    OpOperand,
    FloatType,
    OpResult,
    LoopResultOp,
)
from sparsene.op_gen.opir.ops import (
    SparseOffsetLoadOp,
    CooAtomicFormatLoadOffOp,
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    ValSidxLoadOp,
    BValLoadOp,
    CooAtomicValRestoreOp,
    McoAtomicFormatLoadMaskOp,
    McoAtomicFormatLoadValOp,
    McoAtomicValRestoreOp,
    MmaOp,
    CValLoadOp,
    CValStoreOp,
)
from sympy import Symbol, Indexed, Expr, Integer, Rational
from sparsene.format.format import Direction



def datatype_to_type(datatype: DataType) -> Type:
    match datatype:
        case DataType.FLOAT:
            return FloatType()
        case DataType.INT:
            return IntType()
        case _:
            raise ValueError(f"Unsupported datatype: {datatype}")


def get_array_type(array_def: ArrayDef) -> Type:
    dims = list(array_def.dims)
    return ArrayType(dims, datatype_to_type(array_def.datatype))


def build_int_value_from_expr(
    expr: Any,
    array_defs: ArrayDefCollection,
    name_hint: str = "len",
) -> Value:
    if isinstance(expr, Value):
        return expr
    if isinstance(expr, (Integer, int)):
        return op_builder.build(ConstantOp, int(expr), IntType(), name_hint=name_hint).result
    if isinstance(expr, str):
        return op_builder.lookup_symbol(expr, IntType())
    return translate_sympy_to_ir(expr, array_defs)


def infer_m_dim_len(
    tile: Value,
    array_defs: ArrayDefCollection,
    name_hint: str = "c_m_len",
) -> Value:
    assert isinstance(tile.type, ArrayType), "C tile should be an array"
    assert len(tile.type.dims) > 0, "C tile should have at least one dimension"
    return build_int_value_from_expr(tile.type.dims[0], array_defs, name_hint=name_hint)


def build_c_val_load_slice(
    c_tile_container: Value,
    offset: Value,
    array_defs: ArrayDefCollection,
    name_hint: str = "c_slice",
) -> Value:
    assert isinstance(c_tile_container.type, ArrayType), "C tile container should be an array"
    slice_dims = list(c_tile_container.type.dims[1:])
    assert len(slice_dims) > 0, "C tile slicing should keep at least one dimension"

    load_len = build_int_value_from_expr(
        slice_dims[0],
        array_defs,
        name_hint=f"{name_hint}_len",
    )

    c_val_load_op = op_builder.build(
        CValLoadOp,
        mem="S2R",
        array=c_tile_container,
        offset=offset,
        length=load_len,
    )
    c_slice_result = OpResult(
        type=ArrayType(slice_dims, c_tile_container.type.datatype),
        defining_op=c_val_load_op,
        result_idx_in_owner=0,
        name_hint=name_hint,
    )
    c_val_load_op.add_result(c_slice_result)
    return c_slice_result

def generate_dispatch(
    schedule: Schedule,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value],
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    """       ，       C Tile Value"""
    if isinstance(schedule, DenseAxisIterator):
        return generate_from_dense_axis_iterator(
            schedule, array_defs, format, current_c_tile, parent_schedule
        )
    elif isinstance(schedule, SparseAxisIterator):
        return generate_from_sparse_axis_iterator(
            schedule, array_defs, format, current_c_tile, parent_schedule
        )
    elif isinstance(schedule, AtomicFormatOp):
        assert current_c_tile is not None, "Atomic format op should receive a C tile from parent schedule"
        #   ：Atomic           ，    parent_schedule
        return generate_from_atomic_format_op(
            schedule, array_defs, format, current_c_tile
        )
    else:
        raise ValueError(f"Unsupported schedule type for dispatch: {type(schedule)}")

def generate_from_computent(computent: Computent) -> MetaOp:
    computent_op = op_builder.build(MetaOp)
    with op_builder.op_scope(computent_op):
        for array_def in computent.array_defs:
            op_builder.lookup_symbol(array_def.name, get_array_type(array_def))
        schedule = computent.schedule
        array_defs = computent.array_defs
        format = computent.format
        if isinstance(schedule, DenseAxisIterator):
            generate_from_dense_axis_iterator(schedule, array_defs, format)
        elif isinstance(schedule, SparseAxisIterator):
            generate_from_sparse_axis_iterator(schedule, array_defs, format)
            # raise NotImplementedError(
            #     "Sparse axis iterator as root schedule is not implemented"
            # )
        elif isinstance(schedule, SparseAxisSlicer):
            raise NotImplementedError(
                "Sparse axis slicer as root schedule is not implemented"
            )
        elif isinstance(schedule, AtomicFormatOp):
            raise NotImplementedError(
                "Atomic format op as root schedule is not implemented"
            )
        else:
            raise ValueError(f"Unsupported schedule type: {type(schedule)}")
    return computent_op


# TODO         active_tile, current_c_tile,       
def generate_from_dense_axis_iterator(
    schedule: DenseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value] = None,
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    l = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
    r = op_builder.build(
        ConstantOp,
        format.get_axis(schedule.axis).length,
        IntType(),
        name_hint="num_row_win",
    ).result

    is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

    #>      （      Direction.ROW，           ）
    active_tile = current_c_tile    
    
    # 1.           C Tile
    #              ，        C_val   （        Tile）
    if not is_reduction and current_c_tile is None and parent_schedule is None:
        current_c_tile = op_builder.lookup_symbol("C_val")

    iter_args = {"C_tile_io": current_c_tile} if current_c_tile else {}

    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(l, r),
        iter_args=iter_args,
    )

    # with op_builder.op_scope(for_loop_op):
    #     op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        
    #     if not is_reduction and active_tile is not None:
    #         #            ，       c_tile
    #         #       c_tile         
    #         #   [BLK_M, N] -> [BLK_M_I, N]
    #         active_tile = op_builder.build(
    #             ArrayRefOp,
    #             active_tile,
    #             # [Symbol(schedule.induction_var)],
    #             [op_builder.lookup_symbol(schedule.induction_var, IntType(), )],
    #             name_hint=f"c_sub_tile",
    #         ).result
        
    #     #>     
    #     inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None

    #     #>       
    #     updated_sub_tile = generate_dispatch(schedule.body, array_defs, format, active_tile, schedule)

    #     #>         ，    sub_tile，   sub_tile  parent tile
    #     if iter_args:
    #         if not is_reduction:
    #             if updated_sub_tile is not None:
    #                 op_builder.build(CValStoreOp, mem="R2S",operands=[updated_sub_tile, inner_c_tile, op_builder.lookup_symbol(schedule.induction_var)])
    #             op_builder.build(LoopResultOp, inner_c_tile) 
    #         else:
    #             op_builder.build(LoopResultOp, updated_sub_tile)
                
    # #                ，        
    # if not is_reduction and parent_schedule is None and for_loop_op.num_results > 0:
    #     op_builder.build(CValStoreOp, mem="R2S", operands=[for_loop_op.get_results()[0]])

    # if iter_args:
    #     return for_loop_op.result
    # return None

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        loop_index = op_builder.lookup_symbol(schedule.induction_var, IntType())
        
        #            
        inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None
        active_sub_tile = inner_c_tile

        # 2.       ，     (Logical Slice)
        if not is_reduction and inner_c_tile is not None:
            active_sub_tile = build_c_val_load_slice(
                inner_c_tile,
                loop_index,
                array_defs,
                name_hint="c_slice",
            )
        
        # 3.     
        #     active_sub_tile        （   ）      （   ）
        updated_res_tile = generate_dispatch(schedule.body, array_defs, format, active_sub_tile, schedule)

        # 4.          
        if iter_args:
            if not is_reduction:
                #          Body              
                if updated_res_tile is not None:
                    store_len = infer_m_dim_len(updated_res_tile, array_defs, name_hint="c_store_len")
                    op_builder.build(CValStoreOp, mem="R2S", operands=[
                        updated_res_tile, 
                        inner_c_tile, 
                        loop_index,
                    ], length=store_len)
                op_builder.build(LoopResultOp, inner_c_tile) 
            else:
                #             
                op_builder.build(LoopResultOp, updated_res_tile)
                
    # #         （  Pass         ，        Store   ）
    # if parent_schedule is None and for_loop_op.num_results > 0:
    #     #          ，        ，      
    #     op_builder.build(CValStoreOp, mem="R2S", operands=[for_loop_op.get_results()[0]])

    return for_loop_op.get_results()[0] if iter_args else None


def translate_sympy_to_ir(expr: Expr, array_defs: ArrayDefCollection) -> Value:
    """  SymPy          OpIR   """
    if str(expr) == "_":
        return Value(type=None, name_hint="_")

    if isinstance(expr, (Integer, int)):
        return op_builder.build(ConstantOp, int(expr), IntType()).result
    
    if isinstance(expr, Rational):
        num = op_builder.build(ConstantOp, int(expr.p), IntType()).result
        den = op_builder.build(ConstantOp, int(expr.q), IntType()).result
        return op_builder.build(DivOp, num, den).result

    if isinstance(expr, Symbol):
        #            Induction Variable      
        return op_builder.lookup_symbol(str(expr), IntType())
    
    if isinstance(expr, Indexed):
        #    val_len[i1]       
        base_name = str(expr.base)
        indices = [translate_sympy_to_ir(idx, array_defs) for idx in expr.indices]
        array_val = op_builder.lookup_symbol(base_name, get_array_type(array_defs[base_name]))
        return op_builder.build(LoadOp, array_val, indices).result

    #       
    if expr.is_Add:
        # args = [translate_sympy_to_ir(a, array_defs) for a in expr.args]
        # res = args[0]
        # for a in args[1:]:
        #     res = op_builder.build(AddOp, res, a).result
        # return res
        # 1.           "_"
        placeholder_args = [a for a in expr.args if str(a) == "_"]
        if placeholder_args:
            # 2.       "_"     
            non_placeholder_args = [a for a in expr.args if str(a) != "_"]
            
            if len(non_placeholder_args) == 0:
                return Value(type=IntType(), name_hint="_")
            
            # 3.           IR（    ，    ）
            base_offset_val = translate_sympy_to_ir(non_placeholder_args[0], array_defs)
            for i in range(1, len(non_placeholder_args)):
                next_val = translate_sympy_to_ir(non_placeholder_args[i], array_defs)
                base_offset_val = op_builder.build(AddOp, base_offset_val, next_val).result
            
            # 4.     ：            
            base_offset_val.name_hint = "slice_offset"
            return base_offset_val
    if expr.is_Mul:
        #>    Mul     ，   sympy   a/b     a * (b^-1)，   opIR    1/b，     a
        # args = [translate_sympy_to_ir(a, array_defs) for a in expr.args]
        # res = args[0]
        # for a in args[1:]:
        #     res = op_builder.build(MulOp, res, a).result
        # return res
        #>    Mul   ，a/b      a/b
        numerators = []
        denominators = []
        
        for arg in expr.args:
            #         : b**-1
            if arg.is_Pow and arg.exp == -1:
                denominators.append(arg.base)
            #            : Rational(1, 32) ->   1,   32
            elif isinstance(arg, Rational):
                if arg.p != 1: numerators.append(Integer(arg.p))
                denominators.append(Integer(arg.q))
            else:
                numerators.append(arg)
        
        #         IR
        if not numerators:
            res_num = op_builder.build(ConstantOp, 1, IntType()).result
        else:
            res_num = translate_sympy_to_ir(numerators[0], array_defs)
            for n in numerators[1:]:
                res_num = op_builder.build(MulOp, res_num, translate_sympy_to_ir(n, array_defs)).result
        
        #       ，         
        if not denominators:
            return res_num
            
        #         IR
        res_den = translate_sympy_to_ir(denominators[0], array_defs)
        for d in denominators[1:]:
            res_den = op_builder.build(MulOp, res_den, translate_sympy_to_ir(d, array_defs)).result
            
        #        ：(n1 * n2 * ...) / (d1 * d2 * ...)
        #     GPU          ，    BLK_K，     
        return op_builder.build(DivOp, res_num, res_den).result
    if expr.is_Pow: #         
        base_val = translate_sympy_to_ir(expr.base, array_defs)
        if expr.exp == -1:
            # 1 / base
            one = op_builder.build(ConstantOp, 1, IntType()).result
            return op_builder.build(DivOp, one, base_val).result
        else:
            exp_val = translate_sympy_to_ir(expr.exp, array_defs)
            return op_builder.build(PowOp, base_val, exp_val).result
        pass #     ，     Mul(a, Pow(b, -1))
    
    #        (   val_len/BLK_K)
    if expr.func.__name__ == 'Mul': #        Rational   Pow(-1)
        #     ：    a / b
        # ...        Sympy        
        pass
    print(expr)
    raise NotImplementedError(f"Sympy expression {expr} translation not implemented.")

def collect_spatial_shapes(node: Schedule, format: Format) -> List[any]:
    """
         schedule  ，          （X /ROW）    
    """
    shapes = []
    curr = node

    while curr is not None:
        if isinstance(curr, (DenseAxisIterator, SparseAxisIterator)):
            axis_obj = format.get_axis(curr.axis)
            if axis_obj.direction == Direction.ROW:
                shapes.append(axis_obj.length)
            curr = curr.body
        elif isinstance(curr, AtomicFormatOp):
            # AtomicFormatOp        ，     
            for axis in curr.axes:
                axis_obj = format.get_axis(axis)
                if axis_obj.direction == Direction.ROW:
                    shapes.append(axis_obj.length)
            break  # AtomicFormatOp     ，    
        else:
            break
    return shapes

def generate_from_sparse_axis_iterator(
    schedule: SparseAxisIterator,
    array_defs: ArrayDefCollection,
    format: Format,
    current_c_tile: Optional[Value] = None,
    parent_schedule: Optional[Schedule] = None,
) -> Value:
    zero = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
    upper_bound = translate_sympy_to_ir(schedule.splen, array_defs)

    is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

    # 1.       （         ，   Spatial            ）
    if is_reduction and current_c_tile is None:
        tile_shape = collect_spatial_shapes(schedule.body, format) + [array_defs["C_val"].dims[-1]]
        current_c_tile = op_builder.build(ConstantOp, 0, ArrayType(tile_shape, FloatType()), name_hint="zero_tile").result
    
    if not is_reduction and current_c_tile is None and parent_schedule is None:
        current_c_tile = op_builder.lookup_symbol("C_val")

    iter_args = {"C_io": current_c_tile} if current_c_tile else {}
    
    for_loop_op = op_builder.build(
        ForLoopOp,
        induction_var=schedule.induction_var,
        range=(zero, upper_bound),
        iter_args=iter_args,
    )

    with op_builder.op_scope(for_loop_op):
        op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        loop_index = op_builder.lookup_symbol(schedule.induction_var, IntType())
        inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None
        
        # 2.      （   Sparse Spatial  ，  CSR     ）
        #             indirect indexing
        active_sub_tile = inner_c_tile
        if not is_reduction and inner_c_tile is not None:
            #         ，          schedule        
            active_sub_tile = build_c_val_load_slice(
                inner_c_tile,
                loop_index,
                array_defs,
                name_hint="c_sparse_slice",
            )

        updated_res_tile = generate_dispatch(schedule.body, array_defs, format, active_sub_tile, schedule)
        
        # 3.       
        if iter_args:
            if not is_reduction:
                #    ：   inner_c_tile[i]
                if updated_res_tile is not None:
                    store_len = infer_m_dim_len(updated_res_tile, array_defs, name_hint="c_sparse_store_len")
                    op_builder.build(CValStoreOp, mem="R2S", operands=[
                        updated_res_tile,
                        inner_c_tile,
                        loop_index,
                    ], length=store_len)
                op_builder.build(LoopResultOp, inner_c_tile)
            else:
                op_builder.build(LoopResultOp, updated_res_tile)

    return for_loop_op.get_results()[0] if iter_args else None

# def generate_from_sparse_axis_iterator(
#     schedule: SparseAxisIterator,
#     array_defs: ArrayDefCollection,
#     format: Format,
#     current_c_tile: Optional[Value] = None,
#     # offset_index: Value,
#     parent_schedule: Optional[Schedule] = None,
# ) -> Value:
#     #> 1.       [l, r)
#     zero = op_builder.build(ConstantOp, 0, IntType(), name_hint="zero").result
#     upper_bound = translate_sympy_to_ir(schedule.splen, array_defs)

#     #> 2.          C （           ，  direction  
#     is_reduction = format.get_axis(schedule.axis).direction == Direction.COL

#     #> 3.       ，      C_tile，     （       Y   ）
#     iter_args = {}
#     if is_reduction:
#         if current_c_tile is None:
#             spatial_shapes = collect_spatial_shapes(schedule.body, format)
#             assert len(spatial_shapes) > 0, "At least one spatial shape should be collected for reduction axis"
#             c_def = array_defs["C_val"] #    C          
#             n_length = c_def.dims[-1]
            
#             tile_shape = spatial_shapes + [n_length] #         [BLK_M_I, N]
            
#             current_c_tile = op_builder.build(ConstantOp, 0, ArrayType(tile_shape, FloatType()), name_hint="zero_tile").result
#     #     iter_args["C_tile_in"] = current_c_tile
#     # assert current_c_tile is not None, "current_c_tile should be initialized at this point"
#     iter_args = {"C_in": current_c_tile} if current_c_tile else {}
#     for_loop_op = op_builder.build(
#         ForLoopOp,
#         induction_var=schedule.induction_var,
#         range=(zero, upper_bound),
#         iter_args=iter_args,
#         name_hint=f"loop_{schedule.induction_var}" if iter_args else None,
#     )

#     with op_builder.op_scope(for_loop_op):
#         op_builder.add_symbol(schedule.induction_var, for_loop_op.get_induction_var())
        
#         #           C Tile   
#         inner_c_tile = for_loop_op.get_iter_arg(0) if iter_args else None

#         updated_sub_tile = generate_dispatch(schedule.body, array_defs, format, inner_c_tile, schedule)
#         # 6.        ，         Tile
#         if iter_args:
#             if not is_reduction:
#                 #            （   CSR     ），    
#                 if updated_sub_tile is not None:
#                      #               
#                      if parent_schedule is not None:
#                          op_builder.build(CValStoreOp, mem="R2S", operands=[updated_sub_tile, inner_c_tile, op_builder.lookup_symbol(schedule.induction_var)])
#                      else:
#                          op_builder.build(CValStoreOp, mem="R2S", operands=[updated_sub_tile, op_builder.lookup_symbol(schedule.induction_var)])
#                 op_builder.build(LoopResultOp, inner_c_tile)
#             else:
#                 #    ：     
#                 op_builder.build(LoopResultOp, updated_sub_tile)
            
#     # return for_loop_op.result
#     if iter_args:
#         return for_loop_op.result
#     return None



def generate_from_sparse_axis_slicer(
    schedule: SparseAxisSlicer,
    array_defs: ArrayDefCollection,
    format: Format,
    offset_index: Value,
    parent_schedule: Optional[Schedule] = None,
) -> None:
    raise NotImplementedError("Sparse axis slicer is not implemented")


def generate_from_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    match schedule.type:
        case AtomicFormatType.DENSE:
            return generate_from_dense_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.COO:
            return generate_from_coo_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )
        case AtomicFormatType.MCO:
            return generate_from_mco_atomic_format_op(
                schedule, array_defs, format, C_val_tile
            )


def build_array_def(
    array_def: ArrayDef,
) -> Value:
    return op_builder.lookup_symbol(
        array_def.name,
        get_array_type(array_def),
    )


def build_array_ref_indices(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    name_hint: Optional[str] = None,
    tile_context: Optional[Dict[Direction, Expr]] = None,
) -> Sequence[Value]:
    indices: List[Value] = []

    #       ，       
    adef = array_defs[array_ref.array]
    for i, index in enumerate(array_ref.indices):
        if isinstance(index, ArrayRef):
            index_result = build_array_ref(
                index,
                array_defs,
                format,
                name_hint=f"{name_hint}_sidx_{i}" if name_hint else None,
            )
        elif isinstance(index, Expr):
            index_result = translate_sympy_to_ir(index, array_defs)
        #>            ，       ，     %_ = external @_: int
        elif index == "_":
            #         Value，   op_builder      Op
            index_result = Value(type=IntType(), name_hint="_")
        else:
            index_result = op_builder.lookup_symbol(
                index,
                IntType(),
            )

        #         Arange  
        hint = getattr(index_result, "name_hint", None)
        if hint == "slice_offset":
            axis_name = adef.axes[i]
            if axis_name == "Y":
                axis_dir = Direction.COL
            else:
                axis_dir = format.get_axis(axis_name).direction
            length = tile_context.get(axis_dir) if tile_context else None
            if length:
                index_result = op_builder.build(ArangeOp, index_result, length).result
        
        indices.append(index_result)
    return indices


def build_array_ref(
    array_ref: ArrayRef,
    array_defs: ArrayDefCollection,
    format: Format,
    name_hint: Optional[str] = None,
    tile_context: Optional[Dict[Direction, Expr]] = None,
) -> OpResult:
    array = build_array_def(array_defs[array_ref.array])
    indices = build_array_ref_indices(
        array_ref, array_defs, format, name_hint=name_hint, tile_context=tile_context
    )

    array_ref_op = op_builder.build(
        ArrayRefOp,
        array,
        indices,
        name_hint=name_hint,
    )

    print(f"(debug)build_array_ref: array={array}, indices={indices}, result={array_ref_op.result}")
    return array_ref_op.result


def generate_from_coo_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:

    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating COO atomic format op with args: ", schedule.args)

    #> ori off_array
    # off_array = op_builder.lookup_symbol(
    #     schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    # )
    #> current len_array
    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    idx_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)
    #> origin coo processing logic
    # ll, rr = op_builder.build(
    #     CooAtomicFormatLoadOffOp,
    #     mem="G2R",
    #     operands=[
    #         off_array,
    #         *build_array_ref_indices(schedule.args[0], array_defs, format),
    #     ],
    #     name_hint=["ll", "rr"],
    # ).results

    # print(C_val_tile)
    # print(B_val_tile)

    # coo_idx, coo_range = op_builder.build(
    #     CooAtomicFormatLoadIdxOp,
    #     mem="G2R",
    #     operands=[
    #         idx_array,
    #         *build_array_ref_indices(schedule.args[1], array_defs, format),
    #         ll,
    #         rr,
    #     ],
    #     out_shape=C_val_tile.type.dims[0] * B_val_tile.type.dims[0],  # type: ignore
    #     name_hint=["coo_idx", "coo_range"],
    # ).results

    # coo_val = op_builder.build(
    #     CooAtomicFormatLoadValOp,
    #     mem="G2S",
    #     operands=[
    #         val_array,
    #         *build_array_ref_indices(schedule.args[2], array_defs, format),
    #         ll,
    #         rr,
    #     ],
    #     out_shape=C_val_tile.type.dims[0] * B_val_tile.type.dims[0],  # type: ignore
    #     name_hint="coo_val",
    # ).result

    # A_val_tile = op_builder.build(
    #     CooAtomicValRestoreOp,
    #     mem="S2R",
    #     operands=[
    #         coo_val,
    #         coo_idx,
    #         coo_range,
    #     ],
    #     out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
    #     name_hint="A_val_tile",
    # ).result

    # C_val_tile = op_builder.build(
    #     MmaOp, "R2R", [A_val_tile, B_val_tile, C_val_tile], name_hint="C_val_tile"
    # ).result
    #> current coo processing logic. using len array instead of off array
    coo_len = op_builder.build(
        LoadOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint="coo_len"
    ).result

    coo_idx_indices = build_array_ref_indices(schedule.args[1], array_defs, format)
    coo_val_indices = build_array_ref_indices(schedule.args[2], array_defs, format)

    idx_tile_shape = C_val_tile.type.dims[0] * B_val_tile.type.dims[0]
    coo_idx = op_builder.build(
        CooAtomicFormatLoadIdxOp,
        mem="G2R",
        operands=[
            idx_array,
            *coo_idx_indices,
        ],
        coo_len=coo_len,
        out_shape=idx_tile_shape,
        name_hint="coo_idx",
    ).result

    coo_val = op_builder.build(
        CooAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            val_array,
            *coo_val_indices,
        ],
        coo_len=coo_len,
        out_shape=idx_tile_shape,
        name_hint="coo_val",
    ).result

    A_val_tile = op_builder.build(
        CooAtomicValRestoreOp,
        mem="S2R",
        operands=[
            coo_val,
            coo_idx,
            coo_len, #        coo_len
        ],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="A_val_tile",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [A_val_tile, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result


    return C_val_tile_updated


def generate_from_mco_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    assert len(schedule.args) == 4
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    assert isinstance(schedule.args[2], ArrayRef)
    assert isinstance(schedule.args[3], ArrayRef)
    print("Generating MCO atomic format op with args: ", schedule.args)
    print("(debug)schedule.axes_len", schedule.axes_len)
    print("(debug)schedule.axes", schedule.axes)

    tile_context = {}
    for axis_name, axis_len in zip(schedule.axes, schedule.axes_len):
        axis_dir = format.get_axis(axis_name).direction
        tile_context[axis_dir] = axis_len

    len_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )
    mask_array = op_builder.lookup_symbol(
        schedule.args[1].array, get_array_type(array_defs[schedule.args[1].array])
    )
    val_array = op_builder.lookup_symbol(
        schedule.args[2].array, get_array_type(array_defs[schedule.args[2].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[3], array_defs, format, name_hint="B_val_tile",
        tile_context=tile_context
    )   

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    mco_len = op_builder.build(
        LoadOp,
        len_array,
        build_array_ref_indices(schedule.args[0], array_defs, format),
        name_hint="mco_len"
    ).result

    mco_mask = op_builder.build(
        McoAtomicFormatLoadMaskOp,
        mem="G2R",
        operands=[
            mask_array,
            *build_array_ref_indices(schedule.args[1], array_defs, format),
        ],
        name_hint="mco_mask"
    ).result

    mco_val = op_builder.build(
        McoAtomicFormatLoadValOp,
        mem="G2S",
        operands=[
            val_array,
            *build_array_ref_indices(schedule.args[2], array_defs, format),
        ],
        mco_len=mco_len,
        out_shape=C_val_tile.type.dims[0] * B_val_tile.type.dims[0],  # type: ignore
        name_hint="mco_val",
    ).result

    mco_val_restore = op_builder.build(
        McoAtomicValRestoreOp,
        mem="S2R",
        operands=[
            mco_val,
            mco_mask,
            mco_len,
        ],
        out_shape=(C_val_tile.type.dims[0], B_val_tile.type.dims[0]),
        name_hint="mco_val_restore",
    ).result

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [mco_val_restore, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated
    # raise NotImplementedError("MCO atomic format op is not implemented")


def generate_from_dense_atomic_format_op(
    schedule: AtomicFormatOp,
    array_defs: ArrayDefCollection,
    format: Format,
    C_val_tile: Value,
) -> OpResult:
    assert len(schedule.args) == 2
    assert isinstance(schedule.args[0], ArrayRef)
    assert isinstance(schedule.args[1], ArrayRef)
    print("Generating Dense atomic format op with args: ", schedule.args)

    val_array = op_builder.lookup_symbol(
        schedule.args[0].array, get_array_type(array_defs[schedule.args[0].array])
    )

    B_val_tile = build_array_ref(
        schedule.args[1], array_defs, format, name_hint="B_val_tile"
    )

    assert isinstance(B_val_tile.type, ArrayType)
    assert isinstance(C_val_tile.type, ArrayType)

    val_tile = op_builder.build(
        LoadOp,
        val_array,
        build_array_ref_indices(schedule.args[0], array_defs, format), name_hint="val_tile"
    )

    C_val_tile_updated = op_builder.build(
        MmaOp, 
        "R2R", 
        [val_tile.result, B_val_tile, C_val_tile], 
        name_hint="C_val_tile"
    ).result

    return C_val_tile_updated  
    # raise NotImplementedError("Dense atomic format op is not implemented")
