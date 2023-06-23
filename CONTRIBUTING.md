# Contributing to simDRIFT

First, thank you for your interest in contributing to simDRIFT! Contributions of all varieties and magnitudes are encouraged and valued. Please take a look at the [Table of Contents](#table-of-contents) below for more information about how best to help with this project. To maintain a manageable and functional scientific research library, please ensure you have thoroughly read the relevant section before contributing. The simDRIFT team and community look forward to your contributions. 

If you like the project but don't have the time to contribute, worry not! Feel free to use simDRIFT (with the appropriate citation) in your teaching or research, suggest features you want to see, or let others know about the project. The simDRIFT team greatly appreciates your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Questions](#questions)
- [How To Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Improving The Documentation](#improving-the-documentation)
- [Style](#style)


## Code of Conduct

The [CONTRIBUTING.md Code of Conduct](simDRIFT/main/CODE_OF_CONDUCT.md) applies to everyone involved in this project, regardless of contribution history. By participating, you are expected to uphold this code. Please report unacceptable behavior to contact.simDRIFT@gmail.com.

## Questions

> Should you have a question about simDRIFT, please feel free to ask! However, please note that the simDRIFT team will assume that those asking questions have read the available [documentation]( https://simDRIFT.readthedocs.io/)

Before you ask a question, it is best practice to search for existing [issues](/issues) that may be relevant. If you find an issue that is suitably similar to your question but still need additional clarification, please write your question as a comment on the relevant issue.

If no relevant issue exists, the most effective way to ask your question is as follows:

- Open an [issue](/issues/new).
- Provide as much context as possible about what problem you're running into. Include error messages, if present.
- Provide information regarding what version of the project repository you are using.

We will strive to read and address the issue as soon as possible, then communicate the changes in the appropriate venue. 

## How To Contribute

> ### Legal Notice 
> By contributing to this project, you agree that you are the sole author of the contributed content, that you have the necessary rights to the content, and that the content you contribute may be distributed under the project license.


### Reporting Bugs


#### Before Submitting a Bug Report

A good bug report is a detailed bug report. Therfore, we ask you to investigate the issue, collect detailed information, and provide a high-quality description of the issue you encountered. Completing the following steps before submitting a bug report will help the simDRIFT team quickly address the potential issue(s).

- Make sure that you are using the latest version.
- To the greatest degree possible, ensure that the issue is not caused by a user-side error (e.g., using incompatible environment components/versions – see the [documentation](https://simDRIFT.readthedocs.io/)) If you are looking for support, please see [this section](#questions).
- To see if others have experienced (and potentially already solved) the issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](issues?q=label%3bug).
- Collect information about the bug:
- Stack trace (traceback), including relevant error messages
- OS Platform and Version (e.g., Windows, Linux, macOS, x86, ARM)
- Version information for Python, CUDA, and PyTorch
- If your issue occurs only for specific inputs, please also include your inputs and the resultant output
- Whether the bug is reliably reproducible (with the current version and with older versions)


#### How Do I Submit a Good Bug Report?

> Never report security-related issues, vulnerabilities, or bugs including sensitive information to the issue tracker, or elsewhere in public. Instead, please report potentially security-sensitive bugs directly to the simDRIFT team at contact.simDRIFT@gmail.com.


The simDRIFT team exclusively uses GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [issue](/issues/new). Please do not add a label until a member of the simDRIFT team has had a chance to classify your issue.
- Explain the expected behavior and the behavior you observe.
- Provide as much context as possible, including the *step-by-step process* that someone else can follow to independently reproduce the issue. If you use a script to run simDRIFT, please include a copy of the relevant code. Good bug reports will isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The simDRIFT team will label the issue accordingly.
- A simDRIFT team member will try reproducing the issue using your steps. If there are no reproduction steps, the team will ask you for those steps and label the issue accordingly. Bugs with the `needs clarification` tag will not be addressed until they are reproduced.
- If the team can reproduce the issue, it will be marked `needs-fix` and possibly other tags.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for simDRIFT, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.


#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://simDRIFT.readthedocs.io/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](/issues) to see if the enhancement has already been suggested. Each major feature version will have an associate Feature Request Thread pinned to the Issues page. If your suggestion has already been made, feel free to add a comment or reaction to the existing suggestion to record your opinion.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case for the merits of this feature. Remember that the simDRIFT team focuses on features that will be useful to most of our users (or the scientific community writ large). If you're interested in a feature that only a minority of users will use, consider writing an add-on/plugin library.


#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](/issues).

- Identify the suggestion by using a **clear and descriptive title** for the issue or comment on the Feature Request Thread.
- Provide a **detailed description of the suggested enhancement**. The more detail, the better.
- **Describe** the current behavior of the program and **explain** which functionality you would like to be added. **elaborate** on why. Feel free to mention similar features in other software programs to clarify your point and serve as inspiration.
- **Elaborate** on why you believe this enhancement will be helpful to most contributing members or the broader scientific community.



### Improving The Documentation

This section guides you through submitting improvements to the documentation. Following these guidelines will help maintainers keep clean and consistent documentation.

#### Submitting a Documentation Improvement Suggestion

- Make sure that you have read the latest version of the [simDRIFT documentation](https://simDRIFT.readthedocs.io/).
- Catalog the scope of your suggested change – is it a change specific to one function or entry, or will it affect other documentation entries?
- Determine whether the suggested change requires substantial restructuring or reformatting of the documentation?
- Ensure that you have a concrete change to suggest – nebulous suggestions will be returned for further detail.
- Once you have addressed the points above, please raise a [GitHub issues](/issues) with “Documentation” somewhere in the title. A simDRIFT team member will review the issue and provide the post with a relevant tag. Requests for further clarification and discussion of the suggested change will be posted to the relevant issue as soon as possible. (Note that issues regarding functionality will be prioritized over documentation improvements.)


## Style

simDRIFT is written in a clear and easy-to-follow style. As a team, we strongly prefer clean and human-readable code. Obscure variable names are discouraged. Please do your best to write any contributions in a compatible style to expedite their incorporation into the project. If you are concerned about whether your contribution is written in a compatible style, please feel free to reach out to the simDRIFT team for further clarification.

Thank you, again, for your support of simDRIFT!
-The simDRIFT Team
     Jacob Blum (Washington University in St. Louis)
     Kainen L. Utt, Ph.D. (Washington University in St. Louis)

