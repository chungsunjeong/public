%% Gaussian Process
% Start!

% final AI/DM application
tic;
clear all
close all
%figure
warning('off','all')
set(0,'DefaultAxesFontSize',15)

%% Inputload('emotiv_response_data3.mat')


load('emotiv_response_data.mat')
person=3;
y_No=y_No(:,1);
y_Yes=y_Yes(3:end,1:person);
X=time(3:end);
length_X=person*length(X); 

for i=1:person-1
    X=[X;time(3:end)*(1.000001)^(i)];
end

X=X(:);
XStar=linspace(min(X),max(X),100)';
y1=y_Yes(:);
y2=y_No(:);
XStar=linspace(min(X),max(X),100)';
y=y1;
% useful values
length_XStar=length(XStar);

%% Model selection with Hyperparameters : Marginal likelihood (evidence)
% Assume : sigma_n is constant.
sigma_n=3.5;
%sigma_n=0.02;

% Sample space for two hyperparameters : sigma_f and l
sigma_f_tmp= linspace(4,6,20)';
l_tmp= linspace(20,40,20)';

% useful values
length_sigma_f=length(sigma_f_tmp);
length_l=length(l_tmp);

% Allocation of log-evidence and kernel function
logE = zeros(length_sigma_f,length_l);
K_y=zeros(length_X,length_X);

% sample for log-evidence
for i=1:length_sigma_f
    for j=1:length_l
        for m=1:length_X
            for n=1:length_X
                K_y(m,n) = sigma_f_tmp(i)^2*exp(-1/(2*l_tmp(j)^2)*(X(m)-X(n))^2)+sigma_n^2*(m==n);
            end
        end
        logE(i,j) = -1/2*y'*(K_y)^-1*y-1/2*log(det(K_y))-length_X*log(2*pi)/2;
    end
end
%K_y=gather(temp);

% Find maximum values and index of log-evidence
logE=real(logE);
[max_logE,order]=max(logE(:));
[I_row,I_col]=ind2sub(size(logE),order);

% Finally, we can choose these hyperparameters.
sigma_f = sigma_f_tmp(I_row);
l = l_tmp(I_col);

% Contour plot
figure1=figure;
hold on
grid on
axis tight
contour(sigma_f_tmp,l_tmp,exp(logE'/1000));%(exp(logE)'));%-exp(max_logE))');
plot(sigma_f,l,'ro');
xlabel('\sigma_f')
ylabel('l')


%% conditioning : Inference

% kernel function with previous hyperparameters
for m=1:length_X
    for n=1:length_X
        K_y(m,n) = sigma_f^2*exp(-1/(2*l^2)*(X(m)-X(n))^2)+sigma_n^2*(m==n);
    end
end

% kernel function between XStar and X
KStar=zeros(length_X,length_XStar);
for m=1:length_X
    for n=1:length_XStar
        KStar(m,n) = sigma_f^2*exp(-1/(2*l^2)*(X(m)-XStar(n))^2);
    end
end

% kernel function between XStar and XStar
KStarDouble=zeros(length_XStar,length_XStar);
for m=1:length_XStar
    for n=1:length_XStar
        KStarDouble(m,n) = sigma_f^2*exp(-1/(2*l^2)*(XStar(m)-XStar(n))^2);
    end
end


L=chol(K_y,'lower');
alpha=L'\(L\y);
fStar=KStar'*alpha;
v=L\KStar;
V_fStar=KStarDouble-v'*v;

%% Prediction plot
figure2=figure;
hold on
grid on
axis auto
xlabel('x')
ylabel('y')

h3_figure2=shadedErrorBar(XStar,fStar,[1.96*sqrt(diag(V_fStar)) 1.96*sqrt(diag(V_fStar))],{'-r','LineWidth',3},0.1);
h4_figure2=plot(X,y,'ro');
axis tight
%fStar2=mean(y)+KStar'/(K+sigma_n^2*eye(length(X)))*(y-mean(y));
ylim([-20 60])
xlabel('time [ms]')
ylabel('EEG Amp [\muV]')

% %% Graphical handle
% warning('on','all')
% 
% set(figure1,'Position',[0 400 560 420])
% %set(gca,'FontSize',18);
% set(figure2,'Position',[800 400 560 420])
% %set(gca,'FontSize',16);
% %set(findall(gcf,'type','text'),'FontSize',18)
% %set(findall(gcf,'-property','FontSize'),'FontSize',12)
% set(h1_figure1,'LineWidth',1.5)
% set(h2_figure1,'LineWidth',3)
% set(h3_figure2.mainLine,'LineWidth',2)
% set(h4_figure2,'LineWidth',3)
% legend('Input data : (X,y)','prediction : yStar')
% toc