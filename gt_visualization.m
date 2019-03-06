data = load("sel_annot_list.mat");
data1 = data.sel_annot_list;
num_fr = size(data1,2);
num_kpts = 16;

path = 'C:\Users\liuyi\Desktop\AI_human_colabeling_toolbox\MPII_images\GT_Vis';
mkdir(path);
    
mpii_pairs = [0  1; 1  2; 2  6; 3  6; 3  4; 4  5; 6  7; 7  8; 8  9; 10  11; 11  12; 7  12; 7  13; 13  14; 14  15];

for i = 1:num_fr
    img_name = data1(i).image.name;
    data2 = data1(i).annorect;
    num_poses = length(data2);
    f = fullfile('C:\Users\liuyi\Desktop\AI_human_colabeling_toolbox\MPII_images',img_name);
    img = imread(f);
    fig = figure('name',img_name);
    imshow(img,'InitialMagnification','fit');
    axis off
    hold on

    if num_poses == 0 || isempty(data1(i).vididx)
        close(fig)
        f1 = fullfile('C:\Users\liuyi\Desktop\AI_human_colabeling_toolbox\MPII_images\AI_Vis',img_name);
        f2 = fullfile('C:\Users\liuyi\Desktop\AI_human_colabeling_toolbox\MPII_images\Res_Vis',img_name);
        delete(f1)
        delete(f2)
        continue
    end
    for j = 1:num_poses
        kpts = nan(num_kpts, 2);
        if isfield(data2(j),'annopoints') && ~isempty(data2(j).annopoints)
            pose_x = data2(j).x2;
            pose_y = data2(j).y1;
            text(pose_x,pose_y,int2str(j),'FontSize',12,'FontWeight','bold','HorizontalAlignment','left','Color', 'g')
            data3 = data2(j).annopoints.point;
            num_points = size(data3, 2);
%             fprintf("%d_%d\n",i,j)
            for k = 1:num_points  
                x = data3(k).x;
                len_x = length(x);
                y = data3(k).y;
                %scatter(x,y,'filled')
                plot(x,y, 'yo', 'LineWidth', 1, 'MarkerSize', 2, 'MarkerFaceColor','y');
                text(x,y,int2str(data3(k).id),'FontSize',8,'HorizontalAlignment','left','Color', 'r')
                kpts(data3(k).id + 1, 1) = x;
                kpts(data3(k).id + 1, 2) = y;
            end    
        else
            continue
        end
        for m = 1:size(mpii_pairs,1)
            partA = mpii_pairs(m,1) + 1;
            partB = mpii_pairs(m,2) + 1;
            if ~isnan(kpts(partA, 1)) && ~isnan(kpts(partB, 1))
                plot([kpts(partA, 1) kpts(partB, 1)],[kpts(partA, 2) kpts(partB, 2)],'c','LineWidth',1);
            end
        end
    end
    hold off
    F = getframe;
    dis_file = fullfile(path,img_name);
    imwrite(F.cdata, dis_file)
    close(fig)
end