"""The same model construction for training and checkpoint evaluation."""

def build_model(n_params, args, norm_mat, si_norm_mat, train_sp_mat):
    if args.gnn == 'lightgcn':
        from modules.LightGCN import LightGCN
        return LightGCN(n_params, args, norm_mat)
    if args.gnn == 'igcn':
        from modules.LightGCN_StabCF import StabCF2
        return StabCF2(n_params, args, norm_mat, si_norm_mat)
    from modules.LightGCN import (SimGCL, XSimGCL, SGL, XSGL, RecDCL, XRecDCL,
                                 XLightGCN, AHNS, XAHNS, DirectAU, XDirectAU, GraphAU, XGraphAU)
    models = dict(simgcl=SimGCL, xsimgcl=XSimGCL, sgl=SGL, xsgl=XSGL,
                  recdcl=RecDCL, xrecdcl=XRecDCL, xlightgcn=XLightGCN, ahns=AHNS,
                  xahns=XAHNS, directau=DirectAU, xdirectau=XDirectAU,
                  graphau=GraphAU, xgraphau=XGraphAU)
    if args.gnn not in models:
        raise NotImplementedError('unknown gnn type: ' + args.gnn)
    inputs = [n_params, args, norm_mat]
    if args.gnn.startswith('x'):
        inputs.append(si_norm_mat)
    if args.gnn in ('sgl', 'xsgl'):
        inputs.append(train_sp_mat)
    return models[args.gnn](*inputs)
