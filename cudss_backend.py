"""Optional cuDSS SPD factor-once backend; only solve enters the CUDA graph."""
import ctypes as ct
import importlib.metadata
import time
import numpy as np
import warp as wp

class CuDSS:
    def __init__(self, matrix, rhs, solution, device='cuda:0'):
        p = importlib.metadata.distribution('nvidia-cudss-cu12').locate_file('nvidia/cu12/lib/libcudss.so.0')
        self.lib = ct.CDLL(str(p))
        P,I,L,S = ct.c_void_p,ct.c_int,ct.c_int64,ct.c_size_t
        signatures = {
            'Create':[ct.POINTER(P)], 'Destroy':[P], 'SetStream':[P,P],
            'ConfigCreate':[ct.POINTER(P)], 'ConfigDestroy':[P],
            'DataCreate':[P,ct.POINTER(P)], 'DataDestroy':[P,P],
            'ConfigSet':[P,I,P,S], 'DataGet':[P,P,I,P,S,ct.POINTER(S)],
            'MatrixCreateDn':[ct.POINTER(P),L,L,L,P,I,I],
            'MatrixCreateCsr':[ct.POINTER(P),L,L,L,P,P,P,P,I,I,I,I,I,I],
            'MatrixDestroy':[P], 'Execute':[P,I,P,P,P,P,P],
        }
        for name,args in signatures.items():
            f=getattr(self.lib,'cudss'+name);f.argtypes=args;f.restype=I
        self.handle,self.config,self.data,self.a,self.b,self.x = [P() for _ in range(6)]
        self.call('Create',ct.byref(self.handle))
        self.call('SetStream',self.handle,wp.get_stream(device).cuda_stream)
        self.call('ConfigCreate',ct.byref(self.config))
        self.call('DataCreate',self.handle,ct.byref(self.data))
        # Disable refinement: independently measured residuals expose precision limits.
        ir=I(0);self.call('ConfigSet',self.config,5,ct.byref(ir),ct.sizeof(ir))
        self.rows=wp.array(matrix.indptr.astype(np.int32),device=device)
        self.cols=wp.array(matrix.indices.astype(np.int32),device=device)
        self.values=wp.array(matrix.data.astype(np.float32),device=device)
        self.rhs,self.solution=rhs,solution
        self.device=device
        self.metadata_constants={}
        self.patched_nodes=0
        n=matrix.shape[0]
        self.call('MatrixCreateCsr',ct.byref(self.a),n,n,matrix.nnz,self.rows.ptr,None,
                  self.cols.ptr,self.values.ptr,10,10,0,3,0,0)
        self.call('MatrixCreateDn',ct.byref(self.b),n,1,n,rhs.ptr,0,0)
        self.call('MatrixCreateDn',ct.byref(self.x),n,1,n,solution.ptr,0,0)
        wp.synchronize_device(device);start=time.perf_counter()
        self.call('Execute',self.handle,3,self.config,self.data,self.a,self.x,self.b)
        self.call('Execute',self.handle,4,self.config,self.data,self.a,self.x,self.b)
        wp.synchronize_device(device)
        self.factor_seconds=time.perf_counter()-start
        nnz=L();written=S()
        self.call('DataGet',self.handle,self.data,1,ct.byref(nnz),ct.sizeof(nnz),ct.byref(written))
        self.factor_nnz=nnz.value
        info=I();written=S()
        self.call('DataGet',self.handle,self.data,0,ct.byref(info),ct.sizeof(info),ct.byref(written))
        if info.value: raise RuntimeError(f'cuDSS factorization info={info.value}')
        # Discover immutable metadata outside the trajectory and outside capture.
        self.call('Execute',self.handle,1008,self.config,self.data,self.a,self.x,self.b)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            self.call('Execute',self.handle,1008,self.config,self.data,self.a,self.x,self.b)
        self._device_pointer_nodes(capture.graph.graph.value,prepare=True)
    def call(self,name,*args):
        status=getattr(self.lib,'cudss'+name)(*args)
        if status: raise RuntimeError(f'cuDSS {name}: status {status}')
    def __call__(self):
        self.call('Execute',self.handle,1008,self.config,self.data,self.a,self.x,self.b)
        if wp.get_device(self.device).is_capturing:
            self._device_pointer_nodes()
    def _device_pointer_nodes(self,graph=None,prepare=False):
        # cuDSS 0.8 refreshes fixed dense-vector metadata using small HtoD
        # copies. Conditional bodies forbid host transfers. Snapshot these at
        # setup and substitute DtoD sources using public CUDA graph APIs.
        # Every later capture must match both destination and exact bytes.
        from cuda.bindings import runtime as cu
        def checked(result):
            if result[0] != cu.cudaError_t.cudaSuccess:raise RuntimeError(str(result[0]))
            return result[1:]
        if graph is None:
            info=checked(cu.cudaStreamGetCaptureInfo(wp.get_stream(self.device).cuda_stream))
            graph=info[2]
        _,count=checked(cu.cudaGraphGetNodes(graph))
        nodes,_=checked(cu.cudaGraphGetNodes(graph,count))
        for node in nodes:
            kind,=checked(cu.cudaGraphNodeGetType(node))
            if kind != cu.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:continue
            params,=checked(cu.cudaGraphMemcpyNodeGetParams(node))
            if params.kind != cu.cudaMemcpyKind.cudaMemcpyHostToDevice:continue
            if (params.extent.width,params.extent.height,params.extent.depth)!=(8,1,1):
                raise RuntimeError('Unexpected cuDSS host transfer size')
            contents=ct.string_at(int(params.srcPtr.ptr),8)
            key=int(params.dstPtr.ptr)
            if prepare:
                self.metadata_constants[key]=(contents,wp.array(np.frombuffer(contents,dtype=np.uint8).copy(),device=self.device))
            if key not in self.metadata_constants or contents!=self.metadata_constants[key][0]:
                raise RuntimeError('cuDSS metadata changed after setup; refusing to capture')
            params.srcPtr.ptr=self.metadata_constants[key][1].ptr
            params.kind=cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice
            checked(cu.cudaGraphMemcpyNodeSetParams(node,params))
            self.patched_nodes+=1
    def close(self):
        for name,args in [('MatrixDestroy',(self.a,)),('MatrixDestroy',(self.b,)),('MatrixDestroy',(self.x,)),
                          ('DataDestroy',(self.handle,self.data)),('ConfigDestroy',(self.config,)),('Destroy',(self.handle,))]:
            if args[-1].value: self.call(name,*args);args[-1].value=None
