from matplotlib import pyplot as pp


def crm(a, b, c):
    return a + 0.6 * (b + 0.7 * (c + 0.3 * 40))


def mrc(a, b, c):
    return a + 0.3 * (b + 0.7 * (c + 0.6 * 40))


def rcm(a, b, c):
    return a + 0.7 * (b + 0.6 * (c + 0.3 * 40))


def mcr(a, b, c):
    return a + 0.3 * (b + 0.6 * (c + 0.7 * 40))


def cmr(a, b, c):
    return a + 0.6 * (b + 0.3 * (c + 0.7 * 40))


def rmc(a, b, c):
    return a + 0.7 * (b + 0.3 * (c + 0.6 * 40))


a_crm = []
a_mrc = []
a_rcm = []
a_mcr = []
a_cmr = []
a_rmc = []


def c_cmr(c, m, r):
    print("C ", c, "M ", m, "R ", r)
    print("CRM ", crm(c, r, m))
    print("MRC ", mrc(m, r, c))
    print("RCM ", rcm(r, c, m))
    print("MCR ", mcr(m, c, r))
    print("CMR ", cmr(c, m, r))
    print("RMC ", rmc(r, m, c))

    a_crm.append(crm(c, r, m))
    a_mrc.append(mrc(m, r, c))
    a_rcm.append(rcm(r, c, m))
    a_mcr.append(mcr(m, c, r))
    a_cmr.append(cmr(c, m, r))
    a_rmc.append(rmc(r, m, c))


for i in range(0, 375000, 1):
    c_cmr(-0.5, -5, -(i / 1000000))
pp.plot(a_crm)
pp.plot(a_mrc)
pp.plot(a_rcm)
pp.plot(a_mcr)
pp.plot(a_cmr)
pp.plot(a_rmc)
pp.legend(["CRM","MRC","RCM","MCR","CMR","RMC"])
#pp.plot(a_crm, a_mrc, a_rcm, a_mcr, a_cmr, a_rmc])
pp.show()
