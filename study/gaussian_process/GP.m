tic;
clear
color={'-b' '-r'};
load('emotiv_response_data.mat')
%%
y_No=y_No(:,:);
y_Yes=y_Yes(:,:);
X=time;
X=repmat(X,1,size(y_No,2));
X=X(:);
XStar=linspace(min(X),max(X),100)';
y1=y_Yes(:);
y2=y_No(:);

%%
sigma1 = 0.1;
sigma2 = 0.1;
kparams1=[5, 5];
kparams2=kparams1;
gprMdl=fitrgp(X,y1,'KernelFunction','squaredexponential',...
    'KernelParameters',kparams1,'Sigma',sigma1);
gprMdl2=fitrgp(X,y2,'KernelFunction','squaredexponential',...
    'KernelParameters',kparams2,'Sigma',sigma2);

[ypred, ystd, yint]=predict(gprMdl,XStar);
[ypred2, ystd2, yint2]=predict(gprMdl2,XStar);
f=figure;
grid on
hold on
f1=shadedErrorBar(XStar,ypred,[yint(:,2)-ypred ypred-yint(:,1)],{'-r','LineWidth',3},0.1);
f2=shadedErrorBar(XStar,ypred2,[yint2(:,2)-ypred2 ypred2-yint2(:,1)],{'-b','LineWidth',3},0.1);
f3=plot(XStar,ypred,'-r','LineWidth',3,'DisplayName','"Yes" Data GP prediction');
f4=plot(XStar,ypred2,'-b','LineWidth',3,'DisplayName','"No" Data GP prediction');
axis tight
ylim([-20 60])
xlabel('time [ms]')
ylabel('EEG Amp [\muV]')
legend([f3 f4])
toc