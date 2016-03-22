from dolfin import *
import numpy as np
import scipy.sparse as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from datetime import date
import os
import math
import sys
import h5py
import ipdb

#########
## generated plots for model_new
#########
# drt = "./results/modelNew2_lin/2015-09-29(whole)"
# drt2 = "./results/modelNew2_lin/2015-09-29(whole)"
drt =  "./results/modelNew2/2016-03-07(whole_3)"
# drt = "./results/modelNew2/2016-03-01(local03)"
drt2 = drt

mesh = Mesh("./test_geo/test2.xml")
subdomains = MeshFunction("size_t", mesh,
                          "./test_geo/test2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh,
                          "./test_geo/test2_facet_region.xml")

V0 = FunctionSpace(mesh,"DG",0)
V = VectorFunctionSpace(mesh,"CG",2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
T = FunctionSpace(mesh, "CG", 1)

t_final = 300
# ##############
# t_final = t_final/3 # only works for MPC
# ##############
dt = 10
time_axis = range(0,t_final+dt,dt)
time_axis = np.array(time_axis)
def retrieve_result( filename_lin, filename_final ):
    fdata = h5py.File( filename_lin, "r" )
    n_f = fdata[ "n_f" ].value
    n_t = fdata[ "n_t" ].value
    n_u = fdata[ "n_u" ].value
    n_p = fdata[ "n_p" ].value
    num_t = fdata[ "num_t" ].value
    num_u = fdata[ "num_u" ].value
    num_p = fdata[ "num_p" ].value
    n_e1 = fdata[ "n_e1" ].value
    n_e2 = fdata[ "n_e2" ].value
    n_e3 = fdata[ "n_e3" ].value
    t_range = fdata[ "t_range" ].value
    v_range = fdata[ "v_range" ].value
    p_range = fdata[ "p_range" ].value
    vbc_point = fdata[ "vbc_point" ].value
    vbc_point2 = fdata[ "vbc_point2" ].value
    vbc2_point = fdata[ "vbc2_point" ].value
    vbc2_point2 = fdata[ "vbc2_point2" ].value
    tq_point = fdata[ "tq_point" ].value
    tq_point2 = fdata[ "tq_point2" ].value
    tq_point3 = fdata[ "tq_point3" ].value
    
    # ipdb.set_trace()
    final_array = np.load( filename_final )

    return ( n_f, n_t, n_u, n_p,
             num_t, num_u, num_p,
             n_e1, n_e2, n_e3,
             t_range, v_range, p_range,
             vbc_point, vbc_point2, vbc2_point, vbc2_point2,
             tq_point, tq_point2, tq_point3,
             final_array )


( n_f, n_t, n_u, n_p,
  num_t, num_u, num_p,
  n_e1, n_e2, n_e3,
  t_range, v_range, p_range,
  vbc_point, vbc_point2, vbc2_point, vbc2_point2,
  tq_point, tq_point2, tq_point3,
  final_array ) = retrieve_result( "model_new2_lin.data",
                                    (drt + "/results1.npy") )

final_array2 = np.load( (drt2 + "/results1.npy") )

# #############
# n_f = n_f/3 # for MPC only
# #############
num_lp = 1
n_total = n_f*( num_t+1+1 ) + num_u + num_p + ( 1 + 1 )*2
n_constraint = n_f*n_e1 + n_e2 + n_e3

tidx = np.arange( 0, n_f*num_t ).reshape( ( n_f, num_t ) ) # temperature indx
uidx = ( tidx.size +
         np.arange( 0, num_u ) ) # velocity indx
pidx = ( tidx.size + uidx.size +
         np.arange( 0, num_p ) ) # pressure indx
vidx = ( tidx.size + uidx.size + pidx.size +
         np.arange( 0, n_f ) ) # heater control, indx
vuidx = ( tidx.size + uidx.size + pidx.size + vidx.size +
          np.arange( 0, 1 ) )   # velocity control 1, indx
vu2idx = ( tidx.size + uidx.size + pidx.size + vidx.size + vuidx.size +
           np.arange( 0, 1 ) )  # velocity control 2, indx
v2idx = ( tidx.size + uidx.size + pidx.size +
          vidx.size + vuidx.size + vu2idx.size +
          np.arange( 0, n_f ) )  # heater control, indx
v2uidx = ( tidx.size + uidx.size + pidx.size +
           vidx.size + vuidx.size + vu2idx.size + v2idx.size +
           np.arange(0,1) )      # velocity control 1 of N2, indx
v2u2idx = ( tidx.size + uidx.size + pidx.size +
           vidx.size + vuidx.size + vu2idx.size +
           v2idx.size + v2uidx.size +
           np.arange(0,1) )      # velocity control 2 of N2, indx

e1idx = np.arange( 0, n_f*n_e1 ).reshape( ( n_f, n_e1 ) )
e2idx = ( e1idx.size +
          np.arange( 0, n_e2 ) )
e3idx = ( e1idx.size + e2idx.size +
          np.arange( 0, n_e3 ) )

tqidx = [] # index for target area
for i in tq_point:
    tqidx.append( t_range.tolist().index(i) )
tqidx = np.array( tqidx )
tq2idx = [] # indx for target area 2
for i in tq_point2:
    tq2idx.append( t_range.tolist().index(i) )
tq2idx = np.array( tq2idx )
tq3idx = []    # indx for target area 3
for i in tq_point3:
    tq3idx.append( t_range.tolist().index(i) )
tq3idx = np.array( tq3idx )

finalT = np.zeros( (n_f+1,n_t) )
for i in range(1,n_f+1):
    finalT[ i,t_range ] = final_array[tidx[i-1,:]]

finalU = np.zeros( (n_u,) )
finalU[v_range] = final_array[uidx]
finalU[vbc_point] = final_array[vuidx]
finalU[vbc_point2] = final_array[vu2idx]
finalU[vbc2_point] = final_array[v2uidx]
finalU[vbc2_point2] = final_array[v2u2idx]

finalP = np.zeros( (n_p,) )
finalP[p_range] = final_array[pidx]

# finalV = np.zeros( (n_f+1,) )
finalV = 1000.0*final_array[vidx]
finalV2 = 1000.0*final_array[v2idx]
final2V = 1000.0*final_array2[vidx]
final2V2 = 1000.0*final_array2[v2idx]

finalVU = final_array[vuidx]
finalVU2 = final_array[vu2idx]
finalV2U = final_array[v2uidx]
finalV2U2 = final_array[v2u2idx]

eng_p = finalP.max()
eng_f1 = eng_p * 2.0/0.1 * t_final**2 * (finalVU**2 + finalVU2**2)**0.5 
eng_h1 = np.sum(finalV) * dt
eng_f2 = eng_p * 2.0/0.1 * t_final**2 * (finalV2U**2 + finalV2U2**2)**0.5 
eng_h2 = np.sum(finalV2) * dt
# tem = np.mean( final_array[ tidx[ 0:n_f/3, tqidx ] ] ) + np.mean( final_array[ tidx[ n_f/3:2*n_f/3, tq2idx ] ] ) + np.mean( final_array[ tidx[ 2*n_f/3:, tq3idx ] ] )
# tem = tem/3
# import ipdb; ipdb.set_trace()

# plot controls for the two cases
'''
plt.figure()
heat1_moving = np.zeros( (n_f+1,) )
heat1_moving[1:] = finalV
heat1_moving[0] = finalV[0]
heat2_moving = np.zeros( (n_f+1,) )
heat2_moving[1:] = finalV2
heat2_moving[0] = finalV2[0]
heat1_whole = np.zeros( (n_f+1,) )
heat1_whole[1:] = final2V
heat1_whole[0] = final2V[0]
heat2_whole = np.zeros( (n_f+1,) )
heat2_whole[1:] = final2V2
heat2_whole[0] = final2V2[0]
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
line1, = plt.step(time_axis,heat1_moving, color='b')
line2, = plt.step(time_axis,heat2_moving,color='b',linestyle="--")
line3, = plt.step(time_axis,heat1_whole,color='r')
line4, = plt.step(time_axis,heat2_whole,color='r',linestyle='--')
plt.xlabel('Time (s)')
plt.ylim(0.0,300)
plt.grid()
plt.savefig((drt + '/linear_heat.pdf'), dpi=1000, format='pdf')
plt.close()
# import ipdb; ipdb.set_trace()
'''
# plot velocity in matplot
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True

########################
contalpha = 0.5
wallthick = 0.5
wallalpha = 0.25
wallcolor = '#2e3436'
heateralpha = 0.4
heatercolor = '#3465A4'

omegazdict = { 'width': 2,
               'height': 2,
               'boxstyle': patches.BoxStyle('Round', pad=0.15),
               'linewidth': 1.0,
               'color': 'black',
               'zorder': 15,
               'fill': False }
heaterdict = { 'width': 1,
               'height': 1,
               'boxstyle': patches.BoxStyle('Round',pad=0.15),
               'linewidth': 1.0,
               'edgecolor': 'black',
               'alpha': heateralpha,
               'facecolor': heatercolor,
               'zorder': 5,
               'fill': True }
walldict = { 'fill': True,
             'color': wallcolor,
             'linewidth': 0,
             'zorder': 5,
             'alpha': wallalpha }
#############

XU = V.dofmap().tabulate_all_coordinates(mesh)
v_dim = V.dim()
XU.resize((V.dim(),2))
xu_cor = XU[::2,0]
# xv_cor = XU[1::2,0]
yu_cor = XU[::2,1]
# yv_cor = XU[1::2,1]
dx = 0.3
dy = 0.3
( xm, ym ) = np.meshgrid( np.arange( xu_cor.min(), xu_cor.max(), dx ),
                          np.arange( yu_cor.min(), yu_cor.max(), dy ) )
# linear interplation
u_x = finalU[::2]
u_y = finalU[1::2]
ipdb.set_trace()
for i in range( len( u_x ) ):
    u_x[i] = np.sign( u_x[i] ) * abs( u_x[i] )**(0.7)
    u_y[i] = np.sign( u_y[i] ) * abs( u_y[i] )**(0.7)
Ux = scipy.interpolate.Rbf(xu_cor, yu_cor, u_x, function='linear')
Uy = scipy.interpolate.Rbf(xu_cor, yu_cor, u_y, function='linear')
u_xi = Ux(xm, ym)
u_yi = Uy(xm, ym)
# speed = np.sqrt( u_xi*u_xi + u_yi*u_yi )
( fig, ax ) = plt.subplots( num = 1,
    # figsize=(6,3),
    dpi=150 )

q_plot = plt.quiver( xm, ym, u_xi, u_yi, pivot = 'tip', color = 'b' )
# plt.streamplot(yu_cor, xu_cor, v_y, u_x)
# plt.colorbar()
q_plot.ax.axes.get_xaxis().set_visible(False)
q_plot.ax.axes.get_yaxis().set_visible(False)
# qk = plt.quiverkey(q_plot, 0.1, 0.1, 0.1,
#                    r'$0.1 \frac{m}{s}$',
#                    fontproperties={'weight': 'bold', 'size':20} )

###########
## omega_z
# ax.add_patch( patches.FancyBboxPatch( xy=(1.5, 2.25), ## bottom-left corner
#                                       **omegazdict ) )
## heaters
# ax.add_patch( patches.FancyBboxPatch( xy=(0.75,3.25), ##bottom-left corner
#                                       **heaterdict ) )
# ax.add_patch( patches.FancyBboxPatch( xy=(8.25,3.25), ##bottom-left corner
#                                       **heaterdict ) )
## walls
ax.add_patch( patches.Rectangle( xy=(0,wallthick), ##bottom-left corner
                                 width=wallthick,
                                 height=4-wallthick,
                                 **walldict ) )
ax.add_patch( patches.Rectangle( xy=(5.5,1.5), ##bottom-left corner
                                 width=wallthick,
                                 height=3.5-wallthick,
                                 **walldict ) )
ax.add_patch( patches.Rectangle( xy=(10,1), ##bottom-right corner
                                 width=-wallthick,
                                 height=3,
                                 **walldict ) )
ax.add_patch( patches.Rectangle( xy=(0,0), ##bottom-left corner
                                 width=10,
                                 height=wallthick,
                                 **walldict ) )
ax.add_patch( patches.Rectangle( xy=(0,5), ##top-left corner
                                 width=10,
                                 height=-wallthick,
                                 **walldict ) )
ax.axis( 'equal' )
ax.axis( 'off' )
# plt.tight_layout()
# fig.subplots_adjust( left=0.03, bottom=0.05, right=1.0, top=0.95 )
# plt.savefig((drt + '/velocity' + str(num_lp) + '.pdf'), dpi=1000, format='pdf')
plt.savefig(('./results/modelNew2/acc/velocity_wh.pdf'),
            dpi=1000, format='pdf')
plt.show()
# plt.close()
import ipdb; ipdb.set_trace()
# plot temperature in matplot
nx = 100
ny = 100
X = T.dofmap().tabulate_all_coordinates(mesh)
X.resize((T.dim(),2))
x_cor = X[:,0]
y_cor = X[:,1]
xi, yi = np.linspace(x_cor.min(), x_cor.max(), nx+1), np.linspace(y_cor.min(), y_cor.max(), ny+1)
xi, yi = np.meshgrid(xi, yi)
tmp_idx = [30]
finalT = finalT[tmp_idx,:]
levels = MaxNLocator(nbins=15).tick_values(finalT.min(), finalT.max())
cmap = plt.get_cmap('Reds')
for i in range( len( tmp_idx ) ):    
    fig = plt.figure()
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(111, aspect="equal")
    temp_T = finalT[i,:]
    rbf = scipy.interpolate.Rbf(x_cor, y_cor, temp_T, function='linear')
    temp_zi = rbf(xi, yi)
    CS = ax.contourf(xi, yi, temp_zi, levels=levels, cmap=cmap)
    CS2 = ax.contour(CS, levels=CS.levels, colors = 'r', hold='on')
    cbar = fig.colorbar(CS)
    cbar.add_lines(CS2)
    CS.ax.axes.get_xaxis().set_visible(False)
    CS.ax.axes.get_yaxis().set_visible(False)
    CS2.ax.axes.get_xaxis().set_visible(False)
    CS2.ax.axes.get_yaxis().set_visible(False)
    fig.savefig((drt + '/temperature' + str(num_lp) + str(i).zfill(2)+'.pdf'), dpi=1000, format='pdf')
    plt.close()
    
# import ipdb; ipdb.set_trace()
# plot pressure in matplot
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
XQ = P.dofmap().tabulate_all_coordinates(mesh)
XQ.resize((T.dim(),2))
xq_cor = XQ[:,0]
yq_cor = XQ[:,1]
temp_P = finalP
rbf_p = scipy.interpolate.Rbf(xq_cor, yq_cor, temp_P, function='linear')
temp_zi = rbf_p(xi, yi)
cmap = plt.get_cmap('Blues')
levels = MaxNLocator(nbins=15).tick_values(finalP.min(), finalP.max())
CS = plt.contourf(xi, yi, temp_zi, levels=levels, cmap=cmap)
plt.colorbar()
CS.ax.axes.get_xaxis().set_visible(False)
CS.ax.axes.get_yaxis().set_visible(False)
plt.savefig( ( drt + '/pressure' + str(num_lp) + '.pdf' ), dpi=1000, format='pdf' )
plt.close()

# plot control
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
heat_time = np.zeros( (n_f+1,) )
heat_time[1:] = finalV
heat_time[0] = finalV[0]
pl_v, = plt.step( time_axis, abs( heat_time ) )
plt.xlabel('Time (s)')
plt.ylabel('Input (W)')
plt.grid()
plt.savefig((drt + '/heat' + str(num_lp) + '.pdf'), dpi=1000, format='pdf')
plt.close()

# plot control
plt.figure()
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True
plt.rcParams['text.usetex'] = True
heat_time = np.zeros( (n_f+1,) )
heat_time[1:] = finalV2
heat_time[0] = finalV2[0]
pl_v, = plt.step( time_axis, abs( heat_time ) )
plt.xlabel('Time (s)')
plt.ylabel('Input (W)')
plt.grid()
plt.savefig((drt + '/heat2' + str(num_lp) + '.pdf'), dpi=1000, format='pdf')
plt.close()
