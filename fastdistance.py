def l2distance(X, Z=None):
    """
    function D=l2distance(X,Z)

    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: dxn data matrix with n vectors (columns) of dimensionality d
    Z: dxm data matrix with m vectors (columns) of dimensionality d

    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)

    call with only one input:
    l2distance(X)=l2distance(X,X)
    """
    if Z is None:
        n, d = X.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
        D = D1 + repmat(s1.T, n, 1)
        np.fill_diagonal(D, 0)
        D = np.sqrt(np.maximum(D, 0))
    else:
        n, d = X.shape
        m, _ = Z.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
        D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
        D = D1 + repmat(s2, n, 1)
        D = np.sqrt(np.maximum(D, 0))
    return D


def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (rows) of dimensionality d
    # Z: mxd data matrix with m vectors (rows) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #

    if Z is None:
        Z = X
    n,d = X.shape     # dimension of X
    m= Z.shape[0]     # dimension of Z
    D=np.zeros((n,m)) # allocate memory for the output matrix
    S=np.zeros((n,m)) # allocate memory for the S matrix
    R=np.zeros((n,m)) # allocate memory for the R matrix
    G=np.zeros((n,m)) # allocate memory for the G matrix
    
    S = np.transpose(np.diagonal(innerproduct(X)) + np.transpose(S))
    R = np.transpose(np.diagonal(innerproduct(Z))) + R
    G = innerproduct(X,Z) + G
    
    D = np.sqrt(S - (2*G) + R)
    return D
#</GRADED>