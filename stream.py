import numpy as np

import pandas

from scipy import optimize

from matplotlib import pylab as plt

import plotly.express as px
import plotly.graph_objects as go
from bokeh.plotting import figure

from glob import glob

from io import BytesIO
import streamlit as st

st.set_page_config(page_title="Logistic function stuff", page_icon=None, layout="wide",initial_sidebar_state="auto")

st.cache_resource()
def trf(x,t1,t2,t3,t4):
    return t1/(1.+t2*np.exp(-t3*(x-t4)))**(1/t2)

st.cache_resource()
def tostr(x):
    return np.array2string(x, precision=2,
                           separator=' , ',
                           formatter={'float_kind':lambda x: "%.2f" % x})[1:-1]
st.cache_resource()
def fit(func,X,Y,Z):
    T1,T2,T3,ET = 92.,1.6,5.6,1.4
    #p0  = np.asarray([T1,T2,T3,ET])

    smask = np.ones_like(X).astype(bool)
    #smask[3] = False
    #smask[-1] = False

    return optimize.curve_fit(func,X[smask],Y[smask],
                              #sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                              bounds= (0.01,np.inf)
                              #bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                              #p0=p0,
                              #maxfev=5000,
                              )

st.cache_resource()
def plot_sep(X,Y,Z,COLORS,XFIT,P,STD,pnames,imnames,show_error,show_varia):
    fig = plt.figure(figsize=(4,8))
    plt.rcParams['text.usetex'] = True

    plt.rc('axes', axisbelow=True)

    for i in range(len(X)):
        plt.subplot(311+i)
        if show_error == True:
            plt.errorbar(X[i],Y[i],Z[i],fmt="o",markersize=4,capsize=3,color=COLORS[i])
        else:
            plt.plot(X[i],Y[i],"o",markersize=4,color=COLORS[i])
        #plt.plot(X[i],Y[i],"o")

        if show_varia==True:


            en  = np.stack([trf(XFIT,*(P[i][:-1]-STD[i])),trf(XFIT,*(P[i][:-1]+STD[i])),trf(XFIT,*P[i][:-1])])
            en[np.isnan(en)] = 0.
            enP = en.max(0)
            enM = en.min(0)


            plt.fill_between(XFIT,enM,enP,color=COLORS[i],alpha=0.2)


            plt.plot(XFIT,trf(XFIT,*P[i][:-1]),"-",
                     color=COLORS[i],
                     alpha=1,
                     linewidth=1,
                     label=imnames[i].replace("_",""))

        else:
            plt.plot(XFIT,trf(XFIT,*P[i][:-1]),"-",
                     color=COLORS[i],
                     alpha=0.25,
                     linewidth=3,
                     label=imnames[i].replace("_",""))

        plt.grid(alpha=0.4)
        if i<2:
            plt.xticks(ticks=np.linspace(0,X[i][-1],5),labels=[])
        else:
            plt.xticks(ticks=np.linspace(0,X[i][-1],5))



        for n,j in enumerate(tostr(P[i]).split(",")):
            plt.plot([],[], "None",label=pnames[n]+" = "+j)


        plt.ylabel("SD value (-)")
        plt.legend(loc="lower right")
    plt.subplots_adjust(hspace=0)

    plt.xlabel("Incubation time (h)")

    buffered1 = BytesIO()
    plt.savefig(buffered1,format="pdf",transparent=True,bbox_inches="tight")

    st.download_button(label      = "Download pdf",
                        data      = buffered1,
                        file_name = "separate_plots.pdf",
                        mime      ='image/pdf')
    st.pyplot(fig)

st.cache_resource()
def plot_tog(X,Y,Z,COLORS,XFIT,P,STD,pnames,imnames,show_error,show_varia):
    fig = plt.figure(figsize=(5,3.5))
    plt.rcParams['text.usetex'] = True

    plt.rc('axes', axisbelow=True)

    for i in range(len(X)):
        if show_error == True:
            plt.errorbar(X[i],Y[i],Z[i],fmt="o",markersize=4,capsize=3,color=COLORS[i])
        else:
            plt.plot(X[i],Y[i],"o",markersize=4,color=COLORS[i])
        #plt.errorbar(X[i],Y[i],Z[i],fmt="o",markersize=4,capsize=3,color=COLORS[i])
        #plt.plot(X[i],Y[i],"o")

        if show_varia==True:


            en  = np.stack([trf(XFIT,*(P[i][:-1]-STD[i])),trf(XFIT,*(P[i][:-1]+STD[i])),trf(XFIT,*P[i][:-1])])
            en[np.isnan(en)] = 0.
            enP = en.max(0)
            enM = en.min(0)


            plt.fill_between(XFIT,enM,enP,color=COLORS[i],alpha=0.2)


            plt.plot(XFIT,trf(XFIT,*P[i][:-1]),"-",
                     color=COLORS[i],
                     alpha=1,
                     linewidth=1,
                     label=imnames[i].replace("_",""))

        else:
            plt.plot(XFIT,trf(XFIT,*P[i][:-1]),"-",
                     color=COLORS[i],
                     alpha=0.25,
                     linewidth=3,
                     label=imnames[i].replace("_",""))

    #for n,j in enumerate(tostr(P[i]).split(",")):
    #    plt.plot([],[], "None",label=pnames[n]+" = "+j)

    plt.grid(alpha=0.4)
    plt.ylabel("SD value (-)")
    plt.legend(loc="lower right")
    plt.subplots_adjust(hspace=0)

    plt.xlabel("Incubation time (h)")

    buffered2 = BytesIO()
    plt.savefig(buffered2,format="pdf",transparent=True,bbox_inches="tight")

    st.download_button(label      = "Download pdf",
                        data      = buffered2,
                        file_name = "combined_plots.pdf",
                        mime      ='image/pdf')
    st.pyplot(fig)


if __name__ == '__main__':

    st.markdown("# Logistic function Model")

    st.write("This web app can be used to fit and plot logistic models to 2D data sets. A set of samples is presented below. You can upload your own sample on the left.")
    st.write("Here you can fit the following generalized Logistic function (Richards growth curve):")
    st.latex(r'''
             f(t,\alpha,\beta,\gamma,\theta) = \frac{\alpha}{[1+\beta e^{(-\gamma (t-\theta))}]^{1/\beta}}
             ''')



    imfiles  = st.sidebar.file_uploader("Upload TXT, CSV files (max. 3 files)", accept_multiple_files=True)
    imnames  = [i.name for i in imfiles]
    pnames   = [r"$\alpha$",r"$\beta$",r"$\gamma$",r"$\theta$","$r$"]
    if len(imfiles) == 0:
        imfiles = sorted(glob("data/*.txt"))[::-1]
        imnames = [i.split("/")[-1].split(".")[0] for i in imfiles]

    #axisops = st.sidebar.columns(3)

    delcol = st.sidebar.columns(3)
    with delcol[0]:
         delimiter = st.text_input("Data delimiter", value=",")


    DATA = [np.loadtxt(i,delimiter=delimiter) for i in imfiles]

    #for i in DATA:
    #    i.shape

    st.sidebar.markdown("""---""")
    st.sidebar.write("select data columns:")
    axisops = st.sidebar.columns(3)
    with axisops[0]:
        xaxis = st.selectbox("x:", range(DATA[0].shape[-1]),0)
    with axisops[1]:
        yaxis = st.selectbox("y:", range(DATA[0].shape[-1]),1)
    with axisops[2]:
        yerror= st.selectbox("error:", range(DATA[0].shape[-1]),2)

    st.sidebar.markdown("""---""")
    st.sidebar.markdown("model data range:")
    axisops2 = st.sidebar.columns(3)
    with axisops2[0]:
        xstart = st.text_input("x0:",0)
    with axisops2[1]:
        xstop  = st.text_input("x1:", round(DATA[0][-1,0]*1.03,2))
    with axisops2[2]:
        xres   = st.text_input("xn:", 100)


    X = np.asarray([i[:,xaxis] for i in DATA])
    Y = np.asarray([i[:,yaxis] for i in DATA])
    Y[Y<0] = 0
    Z = np.asarray([i[:,yerror] for i in DATA])

    XFIT = np.linspace(float(xstart),float(xstop),int(xres))

    COLORS = ["cornflowerblue","indigo","deeppink"]


    P,Q,YFIT,STD = [],[],[],[]
    for i in range(len(imfiles)):
        p,q  = fit(trf,X[i],Y[i],Z[i])
        Yfit = trf(XFIT,*p)
        DIFF = np.diff(Yfit)/np.diff(XFIT)
        r    = np.max(DIFF,axis=-1)
        P.append(np.hstack([p,[r]]))
        Q.append(q)
        STD.append(np.sqrt(np.diag(q)))
        YFIT.append(Yfit)
    P    = np.asarray(P)
    YFIT = np.asarray(YFIT)
    STD  = np.asarray(STD)

    #YFIT = np.asarray([trf(XFIT,*i) for i in P])
    #DIFF = np.diff(YFIT,axis=-1)/np.diff(XFIT)
    #r = np.max(r,axis=-1)

    st.sidebar.markdown("""---""")
    show_error = st.sidebar.checkbox("Show error bars",True)
    show_varia = st.sidebar.checkbox("Show variance (not finished)",False)




    #DATA[0]
    st.markdown("""---""")

    plotcol = st.columns(2)
    with plotcol[0]:
        st.markdown("## Separate plots")

        plot_sep(X,Y,Z,COLORS,XFIT,P,STD,pnames,imnames,show_error,show_varia)


    with plotcol[1]:
        st.markdown("## Combined plots")

        plot_tog(X,Y,Z,COLORS,XFIT,P,STD,pnames,imnames,show_error,show_varia)



    st.markdown("""---""")
    st.markdown("## Results")

    rescols = st.columns([1,1])
    with rescols[0]:
        st.markdown("### Parameter Table")
        DATAFRAME = pandas.DataFrame(P,imnames)
        DATAFRAME.columns=pnames#[:len(P)+1]
        DATAFRAME

        buffered3 = BytesIO()
        DATAFRAME.to_excel(buffered3, sheet_name='fitting_parameters')
        st.download_button(label      = "Download parameters as Excel",
                            data      = buffered3,
                            file_name = "parameter_table.xls")

    with rescols[1]:
        st.markdown("### Qunatisized Model")
        PFRAME = pandas.DataFrame(YFIT,imnames)
        PFRAME.columns=XFIT
        PFRAME

        buffered4 = BytesIO()
        PFRAME.to_excel(buffered4, sheet_name='Models')
        st.download_button(label      = "Download Models as Excel",
                            data      = buffered4,
                            file_name = "Qunatisized_Model.xls")
    #buffered4 = BytesIO()
    st.markdown("""---""")




    st.markdown("### Standart Diviation ")
    st.markdown("Covariance matrix is a square matrix that displays the variance exhibited by elements of datasets and the covariance between a pair of datasets (In our case the raw data 「points」and the fitted data 「lines」) . Variance is a measure of dispersion and can be defined as the spread of data from the mean of the given dataset. Covariance is calculated between two variables and is used to measure how the two variables vary together.")

    rescols2 = st.columns([1,1.2])
    with rescols2[0]:

        STDFRAME = pandas.DataFrame(STD,imnames)
        STDFRAME.columns=pnames[:-1]
        STDFRAME

        buffered5 = BytesIO()
        STDFRAME.to_excel(buffered5, sheet_name='StandartDiviation')
        st.download_button(label      = "Download STD as Excel",
                            data      = buffered5,
                            file_name = "StandartDiviation.xls")

    with rescols2[1]:

        st.markdown("The standart diviation is computed via the covariance matrix (C) of the estimated parameters as follows:")
        st.latex("""
                 \sigma = \sqrt{diag [C(Y,f)]}
                 """)
        st.markdown("Simple explanation of the covariance matric can be found here: - https://www.cuemath.com/algebra/covariance-matrix/")

    st.markdown("""---""")
    st.markdown("## 説明")
    st.write("The model used above is based on the Richards's curve, which is a generalized version of the Sigmaid function. This curve allows an asymetric shape asis necessary for our case above.")
    explot = st.columns([1,0.2,2])
    #with explot[0]:
    with explot[2]:
        st.latex(r"\text{The growthrate }r\text{ is separated into two separate parameters } \beta \text{ and } \gamma \text{.} ")

        selections = np.asarray(imnames+["None"])
        sample    = st.selectbox("Data Sample",selections,3)
        sind      = np.argwhere(selections==sample)[0][0]

    with explot[0]:
        st.latex(r'''
                 f(t,\alpha,\beta,\gamma,\theta) = \frac{\alpha}{[1+\beta e^{(-\gamma (t-\theta))}]^{1/\beta}}
                 ''')

        if sind<len(selections)-1:
            A,B,C,D,E = P[sind]
            A,B,C,D,E = float(A),float(B),float(C),float(D),float(E)

        else:
            A,B,C,D,E = 100.,0.01,0.5,10.,None
        alpha = st.slider("alpha",0.    ,120.,A)
        beta  = st.slider("beta" ,-1.0  ,2.0  ,B)
        gamma = st.slider("gamma",0.01 ,2.0  ,C)
        theta = st.slider("theta",0.    ,25.1,D)
        YFIT  = trf(XFIT,alpha,beta,gamma,theta)
        YFIT[np.isnan(YFIT)]=0.
        fd    = np.diff(YFIT)/np.diff(XFIT)
        rr    = np.max(fd,axis=-1)
        rx    = np.argmax(fd)
        st.latex(r" \text{growth rate: } \quad r(\beta,\theta) = \max \left(\frac{df}{dt}\right)="+str(round(rr,2)))




    with explot[2]:




        fig = plt.figure(figsize=(5,3.5))
        plt.rcParams['text.usetex'] = True

        plt.rc('axes', axisbelow=True)

        ah = np.argwhere(XFIT.astype("int")==int(theta))[0]

        yd  = alpha/2.
        xs0 = XFIT[rx]-yd/rr
        xs1 = XFIT[rx]+yd/rr
        ys0 = YFIT[rx]-yd
        ys1 = YFIT[rx]+yd

        if sind<len(selections)-1:
            plt.plot(X[sind],Y[sind],"o",color=COLORS[sind])

        plt.text(0.5,YFIT[rx]+2,str(r"$\theta = "+str(round(theta,0))+"$"),color="grey")
        plt.arrow(0,YFIT[rx],XFIT[rx]*0.95,0, head_length=0.7, head_width=3, length_includes_head=True,color="grey")
        plt.plot(XFIT[rx],YFIT[rx],"o",color="grey")
        plt.plot([xs0,xs1],[ys0,ys1],"-",color="cornflowerblue",linewidth=3,alpha=0.5)

        plt.text(XFIT[rx]+0.5,YFIT[rx]-5,str(r"$r="+str(round(rr,2))+"$"),color="grey")
        plt.plot(XFIT,YFIT,"-",
                 color="k",
                 alpha=1,
                 linewidth=1.5,
                 label=imnames[i].replace("_",""))

        #for n,j in enumerate(tostr(P[i]).split(",")):
        #    plt.plot([],[], "None",label=pnames[n]+" = "+j)

        plt.grid(alpha=0.4)
        plt.ylim([-5,120])
        plt.ylabel("SD value (-)")
        plt.legend(loc="lower right")
        plt.subplots_adjust(hspace=0)

        plt.xlabel("Incubation time (h)")
        st.pyplot(fig)


    st.markdown("""---""")
    st.markdown("## Further Reading")
    st.write("more information on the Logistic function can be found here:")
    st.write("- https://en.wikipedia.org/wiki/Generalised_logistic_function")
    st.write("- https://en.wikipedia.org/wiki/Logistic_function")
    st.write("")                #mime      ='image/pdf')
    st.write("We produce parameterized fitting function by using a simple least square approach provided by:")
    st.write("- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html")
    st.write("Boundary conditions for all parameters are [0,inf]")
    st.write("More information on the least square method can be found here: ")
    st.write("- https://en.wikipedia.org/wiki/Least_squares")
    st.write("Further methods are e.g. provided by scikit-learn, e.g.: ")
    st.write("- RANSAC: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html")
    st.write("- HUBER: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor")
    st.write("- Theil-Sen: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor")

    # P1  = go.Scatter(x     = X[i],
    #                  y     = Y[i],
    #                  mode  = 'markers',
    #                  #marker=dict(color='white', size=8),
    #                  name  = imnames[i],
    #                  error_y=dict(type   = 'data', # value of error bar given in data coordinates
    #                               array  = Z[i],
    #                               visible=True)
    #                  )
    # P2 = go.Scatter(x    = XFIT,
    #                 y    = trf(XFIT,*p),
    #                 name = 'sinc(x)'
    #                 )
    # P = [P1,P2]
    # fig = go.Figure(data=P)
    # fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    # st.plotly_chart(fig)


    #p = figure(title       ='simple line example',
    #           x_axis_label='x',
    #           y_axis_label='y')

    #p.line(X[0], Y[0], legend_label='Trend', line_width=2)
    #st.bokeh_chart(p, use_container_width=True)
