# Performs simple whitening (subtract mean, divide by std) on a case-wise
# basis on the given features. Used for depth data, since we'd like each
# depth patch to be zero-mean, and scaled separately (so patches with a
# wider std. don't get more weight).
#
# Assumes these are all the same "type" of feature, so we can whiten them
# all together (use the same mean and std for all features).
#
# Author: Aashiq Muhamed
import numpy as np
import scipy
import scipy.signal
import skimage.transform
import pdb
#import matlab.engine
def caseWiseWhiten(feat,mask) :

    # Don't go below some minimum std. for whitening. This is to make sure that
    # cases with low std's (flat table, etc.) don't get exaggerated too much,
    # distorting appearance.
    
    MINSTD = 10;
    feat = np.squeeze(feat)
    mask = np.squeeze(mask)
    # Ignore masked-out values when computing means and stds
    feat[mask == False] = float('nan');
    
    featMean = np.expand_dims(np.nanmean(feat,1),axis = 1);
    
    feat = feat-featMean;
    
    featStd = np.expand_dims(np.maximum(np.nanstd(feat,1),MINSTD),1)
    featStd[np.isnan(featStd)] = MINSTD
    feat = feat/featStd
    
    # Probably don't want NaNs in the final features, convert them back to 0's
    feat[mask == False] = 0;
    
#    print(feat)
#    if((featMean == float(0)).all()) :
#        feat = float('nan')
#        print(feat)
        
    return (feat,featStd)


# Gets the surface normals for a given depth image. Normalizes them so
# the L2 norm for each point is 1 (MATLAB doesn't do this for us)
#
# Author: Aashiq Muhamed
#You might want to check StrawLab's implementation
def getSurfNorm(D) :

    Nx, Ny, Nz = surfnorm(D);
    
    N = np.zeros((np.size(Nx,0), np.size(Nx,1), 3));
    N[:,:,0] = Nx;
    N[:,:,1] = Ny;
    N[:,:,2] = Nz;

    N = N/np.sqrt(np.sum(N**2,2,keepdims = True));
    return N

def surfnorm(z) :
    #SURFNORM Surface normals.
    #   [Nx,Ny,Nz] = SURFNORM(X,Y,Z) returns the components of the 3-D 
    #   surface normal for the surface with components (X,Y,Z).  The 
    #   normal is normalized to length 1.
    #
    #   [Nx,Ny,Nz] = SURFNORM(Z) returns the surface normal components
    #   for the surface Z.
    #
    #   Without lefthand arguments, SURFNORM(X,Y,Z) or SURFNORM(Z) 
    #   plots the surface with the normals emanating from it.
    #
    #   SURFNORM(AX,...) plots into AX instead of GCA.
    #
    #   SURFNORM(...,'PropertyName',PropertyValue,...) can be used to set
    #   the value of the specified surface property.  Multiple property
    #   values can be set with a single statement.
    #
    #   The surface normals returned are based on a bicubic fit of
    #   the data.  Use SURFNORM(X',Y',Z') to reverse the direction
    #   of the normals.
    
    #   Clay M. Thompson  1-15-91
    #   Revised 8-5-91, 9-17-91 by cmt.
    #   Copyright 1984-2017 The MathWorks, Inc.
    
    
    

    
    m,n = z.shape;
    x,y = np.meshgrid(np.arange(1,n+1),np.arange(1,m+1));

    
    m,n = x.shape;
    
#    if ~isequal(size(y),[m,n])
#        error(message('MATLAB:surfnorm:InvalidInput')); 
#    end
#    if ~isequal(size(z),[m,n])
#        error(message('MATLAB:surfnorm:IncorrectInput')); 
#    end
#    if any([m n]<3), error(message('MATLAB:surfnorm:InvalidZValue')); end
    
    stencil1 = np.expand_dims(np.asarray([1, 0, -1]),axis = 0)/2
    stencil2 =  np.asarray([[-1],[0],[1]])/2;
   # pdb.set_trace()
#    if nargout==0 % If plotting, then scale to match plot aspect ratio.
#       % Determine plot scaling factors for a cube-like plot domain.
#       if isempty(cax)
#           cax = gca;
#       end
#       nextPlot = cax.NextPlot;
#       surf(cax,args{:});
#       a = [get(cax,'xlim') get(cax,'ylim') get(cax,'zlim')];
#       Sx = a(2)-a(1);
#       Sy = a(4)-a(3);
#       Sz = a(6)-a(5);
#       scale = max([Sx,Sy,Sz]);
#       Sx = Sx/scale; Sy = Sy/scale; Sz = Sz/scale;
#       
#       % Scale surface
#       xx = x/Sx; yy = y/Sy; zz = z/Sz;
#    else
    xx = x; yy = y; zz = z;

    
    # Expand x,y,z so interpolation is valid at the boundaries.
    xx = np.vstack((3*xx[0,:]-3*xx[1,:]+xx[2,:], xx,3*xx[m-1,:]-3*xx[m-2,:]+xx[m-3,:]));
    #pdb.set_trace()
    xx = np.hstack((np.expand_dims(3*xx[:,0]-3*xx[:,1]+xx[:,2],axis = 1),xx,np.expand_dims(3*xx[:,n-1]-3*xx[:,n-2]+xx[:,n-3],axis = 1)));
    yy = np.vstack((3*yy[0,:]-3*yy[1,:]+yy[2,:],yy,3*yy[m-1,:]-3*yy[m-2,:]+yy[m-3,:]));
    yy = np.hstack((np.expand_dims(3*yy[:,0]-3*yy[:,1]+yy[:,2],axis = 1),yy,np.expand_dims(3*yy[:,n-1]-3*yy[:,n-2]+yy[:,n-3],axis = 1)));
    zz = np.vstack((3*zz[0,:]-3*zz[1,:]+zz[2,:],zz,3*zz[m-1,:]-3*zz[m-2,:]+zz[m-3,:]));
    zz = np.hstack((np.expand_dims(3*zz[:,0]-3*zz[:,1]+zz[:,2],axis = 1),zz,np.expand_dims(3*zz[:,n-1]-3*zz[:,n-2]+zz[:,n-3],axis = 1)));
    
    rows = np.expand_dims(np.arange(2,m+2),axis = 1);
    cols = np.expand_dims(np.arange(2,n+2),axis = 0);
    
    ax = scipy.signal.convolve2d(xx, np.rot90(stencil1,2), mode='same')
    #pdb.set_trace()
    ax = ax[rows-1,cols-1];
    #scipy.signal.convolve2d(X, np.rot90(H), mode='valid')
    ay = scipy.signal.convolve2d(yy, np.rot90(stencil1,2), mode='same')
    ay = ay[rows-1,cols-1];
    az = scipy.signal.convolve2d(zz, np.rot90(stencil1,2), mode='same')
    az = az[rows-1,cols-1];
    
    bx = scipy.signal.convolve2d(xx, np.rot90(stencil2,2), mode='same') 
    bx = bx[rows-1,cols-1];
    by = scipy.signal.convolve2d(yy, np.rot90(stencil2,2), mode='same') 
    by = by[rows-1,cols-1];
    bz = scipy.signal.convolve2d(zz, np.rot90(stencil2,2), mode='same')
    bz = bz[rows-1,cols-1];
    
    # Perform cross product to get normals
    nx = -(ay*bz - az*by);
    ny = -(az*bx - ax*bz);
    nz = -(ax*by - ay*bx);
    
#    if nargout==0
#        % Set the length of the surface normals
#        mag = sqrt(nx.*nx+ny.*ny+nz.*nz)*(10/scale);
#        d = find(mag==0); mag(d) = eps*ones(size(d));
#        nx = nx ./mag;
#        ny = ny ./mag;
#        nz = nz ./mag;
#        
#        % Normal vector points
#        xc = x; yc = y; zc = z;
#        
#        % Set NextPlot to 'add' so that the line is added to the existing axes.
#        % 'surf' calls 'newplot', so the Figure's NextPlot property will
#        % already be set to 'add' at this point.
#        cax.NextPlot = 'add';
#        
#        % use nan trick here
#        xp = [xc(:) Sx*nx(:)+xc(:) nan([numel(xc) 1])]';
#        yp = [yc(:) Sy*ny(:)+yc(:) nan([numel(xc) 1])]';
#        zp = [zc(:) Sz*nz(:)+zc(:) nan([numel(xc) 1])]';
#        
#        plot3(xp(:),yp(:),zp(:),'r-','parent',cax)
#        
#        % Restore the original value for NextPlot.
#        cax.NextPlot = nextPlot;
#        return
#    end
    
    # Normalize the length of the surface normals to 1.
    mag = np.sqrt(nx*nx+ny*ny+nz*nz);
    eps = 2.220446049250313e-16;
    mag[mag==0] = eps;
    nxout = nx/mag;
    nyout = ny/mag;
    nzout = nz/mag;
    
    return (nxout,nyout,nzout)




# Loads the given instance number from the given directory (containing the
# grasping dataset), and returns the RGB-D data as a 4-channel image, with
# RGB as channels 1-3 and D as channel 4.
#
# Author: Aashiq Muhamed

def graspPCDToRGBDImage(dataDir, fileNum) :
#keyboard
    pcdFile = '{}/pcd{:04d}.txt'.format(dataDir,fileNum);
    imFile = '{}/pcd{:04d}r.png'.format(dataDir,fileNum);
    
    points, imPoints,_ = readGraspingPcd(pcdFile);
    
    I = np.asarray(scipy.misc.imread(imFile),dtype = np.float64);
   
    D = np.zeros((np.size(I,1),np.size(I,0)));
    
    D = np.ndarray.flatten(D.T)
    #pdb.set_trace()
    imPoints = np.asarray(imPoints,dtype = np.int64)
    D[imPoints-1] = np.copy(points[:,2]);

    D = np.reshape(D,(np.size(I,0),np.size(I,1)))
    D = np.expand_dims(D,axis = 2)
    I = np.concatenate((I,D),axis = 2);
    return I


#Interpolates masked data (e.g. the depth channel). Uses MATLAB's
# scattered interpolation functionality to do the interpolation, ignoring
# masked-out points.
#
# Can optionally provide the interpolation method to use, if not given,
# defaults to linear.
#
# Author: Aashiq Muhamed

def interpMaskedData(data, mask, method = 'linear') :
    
    # Default method to linear if not provided
    #if nargin < 3:
     #   method = 'linear';
    
    
    # Don't do anything if everything is masked out
    if not np.any(np.ndarray.flatten(mask)) :
        filled = np.copy(data);
        return filled;
    
    #mask = logical(mask);
    
    # Make a grid for X,Y coords, and pick the masked-in points
    X,Y = np.meshgrid(np.arange(1,np.size(data,1)+1),np.arange(1,np.size(data,0)+1));
    
    #Known points
    Xg = X[mask];
    Yg = Y[mask];
    
    # "Query" points, to be filled
    Xq = X[mask == False];
    Yq = Y[mask == False];
    
    Vg = data[mask];
    
    # Run the interpolation, and read out the query points
    #F = TriScatteredInterp(Xg,Yg,Vg,method);
    Vq = scipy.interpolate.griddata( ( Xg, Yg ),
                      Vg,
                      ( Xq, Yq ),
                      method ,
                      fill_value = float('nan')
                      )
    #Vq = F(Xq,Yq);
    
    # Initialize the returned data with the given data, and replace the
    # masked-out points with their interpolated values.
    filled = np.copy(data);
    
    if Vq.size !=0 :
        filled[mask == False] = Vq;
    
    return filled


# Extract a patch from the given image corresponding to the given rectangle
# corners. Rectangle does not have to be axis-aligned.
#
# rectPoints is a 4x2 matrix, where each row is a point, and the columns
# represent the X and Y coordinates. The line segment from points 1 to 2
# corresponds to one of the gripper plates. 
#
# Author: Aashiq Muhamed
def orientedRGBDRectangle(I,rectPoints) :

    SCALE = 1;
    
    if np.any(np.isnan(rectPoints)):
        I2 = float("nan");
        return I2
    
    
    # Compute the gripper angle based on the orientation from points 1 to 2
    gripAng = np.arctan2(rectPoints[0,1] - rectPoints[1,1],rectPoints[0,0]-rectPoints[1,0]);
    
    # Compute the X,Y coords for the image points, and rotate both them and the
    # rectangle corners to be axis-aligned w/r/t the rectangle
    imX,imY = np.meshgrid(np.arange(1,np.size(I,1)+1),np.arange(1,np.size(I,0)+1));
    
    imPoints = np.hstack((np.expand_dims(np.ndarray.flatten(imX.T),axis = 1),np.expand_dims(np.ndarray.flatten(imY.T),axis = 1)));
    
    alignRot = rotMat2D(gripAng);
    
    rectPointsRot = np.dot(rectPoints,alignRot);
    #pdb.set_trace();
    imPointsRot = np.dot(imPoints,alignRot);
    
    #Find the points from the image which are inside the rectangle. 
    inRect = pointsInAARect(imPointsRot,rectPointsRot);
    inRect = np.asarray(inRect,dtype = bool)
    newPoints = imPointsRot[inRect,:];
    
    newPoints = newPoints- np.amin(newPoints,axis = 0);
    newPoints = newPoints * SCALE + 1;
    #pdb.set_trace()
    # Extract data corresponding to the points inside the rectangle
    t1 = np.copy(int(np.round(max(newPoints[:,1]))))
    t2 = np.copy(int(np.round(max(newPoints[:,0]))))
    I2 = np.zeros((t1,t2,4));
    
   
    #newInd = sub2ind(np.size(I2[:,:,0]),int(round(newPoints[:,1])),int(round(newPoints[:,0])));
    
    for i in range(4) :
        channel = np.copy(I[:,:,i]);
        newChannel = np.zeros((np.size(I2,0),np.size(I2,1)));
        
        channel = np.ndarray.flatten(channel.T)
        #newChannel = np.ndarray.flatten(newChannel.T)
        t1 = np.asarray(np.round(newPoints[:,1]),dtype = np.int64)
        t2 = np.asarray(np.round(newPoints[:,0]),dtype = np.int64)
        #pdb.set_trace()
        
        newChannel[t1-1,t2-1] = channel[inRect]
        

        #newChannel = np.reshape(newChannel,(np.size(I2,0),np.size(I2,1)))
    
        #newChannel(newInd) = channel(inRect);
        I2[:,:,i] = newChannel;

    
    return I2


#Extends an image to the given dimensions. Scales so that at least one
#dimension will exactly match the target dimension.
#The other will either also match or be smaller, and be centered and padded 
#by zeros on either side.
#
#Author: Aashiq Muhamed

def  padImage(I,newR,newC) :

    # Compute ratios of the target/current dimensions
    rRatio = newR/np.size(I,0);
    cRatio = newC/np.size(I,1);
    
    #Use these to figure out which dimension needs padding and resize
    #accordingly, so that one dimension is "tight" to the new size

    [h,w,d] = I.shape
    
    if rRatio < cRatio :
        I = skimage.transform.resize(I,(newR, int(w*newR/h)));
    else :
        I = skimage.transform.resize(I,(int(h*newC/w),newC));

    
    #Place the resized image into the full-sized image
    numR, numC, numDims = I.shape;
    
    rStart = int(round((newR-numR)/2)) ;
    cStart = int(round((newC-numC)/2));
    
    I2 = np.zeros((newR,newC,numDims));
    
    I2[rStart:rStart+numR,cStart:cStart+numC,:] = I;
    
    #Mask out padding
    mask = np.zeros((newR,newC));
    
    mask[rStart:rStart+numR,cStart:cStart+numC] = 1;
    return (I2, mask)

def  padImageMatlab(I,newR,newC,eng) :

    # Compute ratios of the target/current dimensions
    rRatio = newR/np.size(I,0);
    cRatio = newC/np.size(I,1);
    
    #Use these to figure out which dimension needs padding and resize
    #accordingly, so that one dimension is "tight" to the new size

    [h,w,d] = I.shape
    
    if rRatio < cRatio :
        I = np.asarray(eng.imresize(matlab.double(I.tolist()),matlab.double([newR, int(w*newR/h)])));
    else :
        I = np.asarray(eng.imresize(matlab.double(I.tolist()),matlab.double([int(h*newC/w),newC])));

    
    #Place the resized image into the full-sized image
    numR, numC, numDims = I.shape;
    
    rStart = int(round((newR-numR)/2)) ;
    cStart = int(round((newC-numC)/2));
    
    I2 = np.zeros((newR,newC,numDims));
    
    I2[rStart:rStart+numR,cStart:cStart+numC,:] = I;
    
    #Mask out padding
    mask = np.zeros((newR,newC));
    
    mask[rStart:rStart+numR,cStart:cStart+numC] = 1;
    return (I2, mask)

#Extends an image to the given dimensions. Scales so that at least one
#dimension will exactly match the target dimension.
#The other will either also match or be smaller, and be centered and padded 
#by zeros on either side.
#
# Author: Aashiq Muhamed



def padMaskedImage2(I,mask,newR,newC) :

    #Compute ratios of the target/current dimensions
    rRatio = newR/np.size(I,0);
    cRatio = newC/np.size(I,1);
    
    [h,w] = I.shape
    #Use these to figure out which dimension needs padding and resize
    #accordingly, so that one dimension is "tight" to the new size
    if rRatio < cRatio :
        I,Imask = resizeMaskedImage2(I,mask,(newR ,int(w*newR/h)));
    else :
        I,Imask = resizeMaskedImage2(I,mask,(int(h*newC/w),newC));
    
    #Place the resized image into the full-sized image
    numR, numC  = I.shape;
    numDims = 1
    rStart = round((newR-numR)/2) ;
    cStart = round((newC-numC)/2) ;
    
    I2 = np.zeros((newR,newC,numDims))
    
    I2[rStart:rStart+numR,cStart:cStart+numC,:] = np.expand_dims(I,axis = 2);
    
    #Also place the mask, masking out padding
    mask = np.zeros((newR,newC,numDims))
    
    mask[rStart:rStart+numR,cStart:cStart+numC,:] = np.expand_dims(Imask,axis = 2);
    
    return (I2, mask)


def padMaskedImage2Matlab(I,mask,newR,newC,eng) :

    #Compute ratios of the target/current dimensions
    rRatio = newR/np.size(I,0);
    cRatio = newC/np.size(I,1);
    
    [h,w] = I.shape
    #Use these to figure out which dimension needs padding and resize
    #accordingly, so that one dimension is "tight" to the new size
    if rRatio < cRatio :
        I,Imask = resizeMaskedImage2Matlab(I,mask,(newR ,int(w*newR/h)),eng);
    else :
        I,Imask = resizeMaskedImage2Matlab(I,mask,(int(h*newC/w),newC),eng);
    
    #Place the resized image into the full-sized image
    numR, numC  = I.shape;
    numDims = 1
    rStart = round((newR-numR)/2) ;
    cStart = round((newC-numC)/2) ;
    
    I2 = np.zeros((newR,newC,numDims))
    
    I2[rStart:rStart+numR,cStart:cStart+numC,:] = np.expand_dims(I,axis = 2);
    
    #Also place the mask, masking out padding
    mask = np.zeros((newR,newC,numDims))
    
    mask[rStart:rStart+numR,cStart:cStart+numC,:] = np.expand_dims(Imask,axis = 2);
    
    return (I2, mask)

#Given an Nx2 matrix of 2D coordinates in imPoints, and a 4x2 list of
#rectangle corners in rectPoints, returns an Nx1 vector indicating whether
#each image point is inside the given rectangle.
#
#Author: Aashiq Muhamed

def pointsInAARect(imPoints,rectPoints) :

    inRect =( np.sign(imPoints[:,0] - rectPoints[0,0]) != np.sign(imPoints[:,0] - rectPoints[2,0])) & \
    (np.sign(imPoints[:,1] - rectPoints[0,1]) != np.sign(imPoints[:,1] - rectPoints[2,1]));
    
    return inRect 


def readGraspingPcd(fname) :
    #Read PCD data
    #fname - Path to the PCD file
    #data - Nx6 matrix where each row is a point, with fields x y z rgb imX imY. x, y, z are the 3D coordinates of the point, rgb is the color of the point packed into a float (unpack using unpackRGBFloat), imX and imY are the horizontal and vertical pixel locations of the point in the original Kinect image.
    #
    #Author: Aashiq Muhamed
   ## Declare data to be global 
   
   
   # Revisit and remove unecessary variables
    fid = open(fname,'r');
    isBinary = False;
    nPts = 0;
    nDims = -1;
    #line = [];
    form = [];
    headerLength = 0;
    IS_NEW = True;
    data = np.zeros((0,6));
    
    for line in fid :
        
        if (len(line) >=4) and (line[:4]=='DATA'):
#            if len(line) == 11 and (line[5:11],'binary') :
#                isBinary = True;
            break
        
#        headerLength = headerLength + len(line) + 1;
#        
#        if len(line) >= 4 and (line[:4] == 'TYPE') :
#            for t in line.split() :
#                if nDims > -1 and (t == 'F') :
#                    form.append('%f');
#                elif nDims > -1 and t == 'U' :
#                    form.append('%d');
#    
#                nDims = nDims+1;
# 
#
#        if len(line) >= 7 and (line[:7] == 'COLUMNS') :
#            IS_NEW = False;
#            for ig in line.split() :
#                form.append('%f');
#                nDims = nDims+1;
#
#    
#        if len(line) >= 6 and (line[:6] == 'POINTS') :
#            nPts = int(line.split()[-1])


 
 ##Ifnore binary       
#    if isBinary :
#        paddingLength = 4096*np.ceil(headerLength/4096);
#        padding = fread(fid,paddingLength-headerLength,'uint8');
#    
#        
#    if isBinary and IS_NEW :
#       data = np.zeros(nPts,nDims);
#       format = regexp(format,' ','split');
#       for i=1:nPts
#          for j=1:length(format)
#             if strcmp(format{j},'%d') 
#                pt = fread(fid,1,'uint32');
#             else
#                pt = fread(fid,1,'float');
#             end
#             data(i,j) = pt;
#          end
#       end
#   
#        elseif isBinary && ~IS_NEW
#           pts = fread(fid,inf,'float');
#           data = zeros(nDims,nPts);
#           data(:) = pts;
#           data = data';
#        else
    #form.append('\n');
    
    #Unnecessary to use form
    C = np.loadtxt(fid);

    points = np.copy(C[:,:3]);
    rgb = unpackRGBFloat(np.copy(C[:,3]));
    rgb = rgb.T;
    imPoints = np.copy(np.asarray(C[:,4]));

    fid.close()

    return (points,imPoints,rgb)

import ctypes

def unpackRGBFloat(rgbfloatdata) :
    # Unpack RGB float data into separate color values
    # rgbfloatdata - the RGB data packed into Nx1 floats
    # rgb - Nx3 unpacked RGB values
    #
    # Author: Aashiq Muhamed
    
    mask = int('000000FF',16);
    #print(rgbfloatdata)
    #pdb.set_trace()
    rgbfloatdata = np.asarray(rgbfloatdata,dtype = np.float32)
    rgb = rgbfloatdata.view(np.uint32)
    
    r = (rgb >>16)&mask
    r = np.asarray(r,dtype = np.uint8)
    
    g = (rgb >>8)&mask
    g = np.asarray(g,dtype = np.uint8)
    
    b = rgb & mask;
    b = np.asarray(b,dtype = np.uint8)
    rgb = np.asarray([r, g, b])
    
    return rgb


# Removes outliers for masked data. Returned data will be zero-mean
#
# Zeros out points with absolute value > stdCutoff * the std of the nonzero
# values in A, and updates the mask to reflect the removed points.
#
# Author: Aashiq Muhamed

def removeOutliers(A,mask,stdCutoff) :

    A[mask == 0] = float('nan');
    Amean = np.nanmean(A);
    A = A - Amean;
    
    Astd = np.nanstd(A);
    
    mask[np.abs(A) > (stdCutoff * Astd)] = 0;
    
    A[mask == 0] = float('nan');
    A = A - np.nanmean(A);
    A[np.isnan(A)] = 0;
    
    return (A,mask)


# Resize an image which includes a mask. This doesn't change the process of
# resizing the image (we assume it's already been interpolated to fill in
# missing data), but we do have to figure out the resized mask as well.
# 
# Do this by checking how much weight the interpolation gives to the good
# data vs the missing data by resizing the mask and checking it against a
# threshold.
#
# Author: Aashiq Muhamed

def resizeMaskedImage2(D, mask, newSz) :

    D = skimage.transform.resize(D,newSz);
    
    mask = skimage.transform.resize(np.asarray(mask,dtype = np.float64),newSz);
    return (D, mask)

# converts rgb data to yuv  (FW-04-03)

def resizeMaskedImage2Matlab(D, mask, newSz,eng) :
   
    D = np.asarray(eng.imresize(matlab.double(D.tolist()),matlab.double(list(newSz))));
    
    mask = np.asarray(eng.imresize(matlab.double(mask.tolist()),matlab.double(list(newSz))));
    return (D, mask)

def rgb2yuv(src) :
    
    #pdb.set_trace()
    # ensure this runs with rgb images as well as rgb triples
    if(len((src.shape)) > 2) :
        
        # rgb image ([r] [g] [b])
        r = np.copy(np.asarray(src[:,:,0],dtype = np.float64));
        g = np.copy(np.asarray(src[:,:,1],dtype = np.float64));
        b = np.copy(np.asarray(src[:,:,2],dtype = np.float64));
        
    elif(len(src) == 3) :
        
        # rgb triplet ([r, g, b])
        r = float(src[0]);
        g = float(src[1]);
        b = float(src[2]);
        
    else :
        
        # unknown input format
        raise Exception('rgb2yuv: unknown input format');
        
    # convert...
    y = 0.3*r + 0.5881*g + 0.1118*b;
    u = -0.15*r - 0.2941*g + 0.3882*b;
    v = 0.35*r - 0.2941*g - 0.0559*b;
    
    
    # generate output
    if(len(src.shape) > 2) :
        dst = np.zeros((y.shape[0],y.shape[1],3))
        # yuv image ([y] [u] [v])
        dst[:,:,0] = y;
        dst[:,:,1] = u;
        dst[:,:,2] = v;
        
    else :
        
        # yuv triplet ([y, u, v])
        dst = np.hstack(y, u, v);
    
    return dst  

# Makes a 2D rotation matrix correspoing to the given angle in radians.
# Imagine that.
#
# Author: Aashiq Muhamed

def rotMat2D(ang) :

    R = np.asarray([[np.cos(ang), -np.sin(ang)], [np.sin(ang) ,np.cos(ang)]])
    return R


# "Smart" interpolation of masked data.
#
# What this means is we do two passes - first, use linear interpolation to
# fill any points it can. Then, use nearest neighbors to fill in any points
# linear interpolation couldn't figure out.
#
# Author: Aashiq Muhamed

def smartInterpMaskedData(data,mask) :
    
    filled = interpMaskedData(data,mask,'linear');
   
    mask2 = np.isnan(filled);
    
    if(np.any(mask2.ravel())) :
        filled = interpMaskedData(filled,~mask2,'nearest');
        
    return filled  

def smartInterpMaskedDataMatlab(data,mask,eng) :
    
    temp = eng.smartInterpMaskedData(matlab.double(data.tolist()),matlab.double(mask.tolist())) 
    temp = np.asarray(temp)
        
    return temp 
