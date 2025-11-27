from differometor.sparse_setups import sparse_uifo


def test_sparse_setup_build(sparse_setup):
    s, _ = sparse_uifo(
        size=sparse_setup['size'],
        element_array=sparse_setup['element_array'],
        mode='space_modulation'
    )

    for element, etype in sparse_setup['elements']:
        assert element in s.nodes._nodes.keys(), f"element {element} not found!"
        assert s.nodes._nodes[element]['component'] == etype, f"component {element} has type {s.nodes._nodes[element]['component']} instead of {etype}"

    for src, tgt, src_port, tgt_port in sparse_setup['connections']:
        assert (src, tgt) in s.edges._edges.keys(), f"edge {src, tgt} not found!"
        c = s.edges._edges[(src, tgt)]
        assert c['source_port'] == src_port, f"incorrect src port for {src, tgt}: {src_port} != {c['source_port']}"
        assert c['target_port'] == tgt_port, f"incorrect tgt port for {src, tgt}: {tgt_port} != {c['target_port']}"
