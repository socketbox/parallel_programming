kernel void ArrayMult( global const float *dA, global const float *dB, global float *dC, global float *dD )
{
  if( dD )
  {
    int gid = get_global_id( 0 );
    dD[gid] = ( dA[gid] * dB[gid] ) + dC[gid];
  }
  else
  {
    int gid = get_global_id( 0 );
    dC[gid] = dA[gid] * dB[gid];
  }
}
